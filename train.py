from torch import nn
# from model import GraphClassifier - Odstraněno pro kompatibilitu s E3 (podstrčeno přes train.GraphClassifier)
import torch
import os
import json
import argparse
import random
import numpy as np
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("egnn_training.log"),
        logging.StreamHandler()
    ]
)





def load_manifest(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    files = manifest.get('files', [])
    batch_paths = [os.path.join(base_dir, fname) for fname in files]
    return manifest, batch_paths


def iter_graph_batches(manifest_path):
    """Yield graph lists one serialized batch at a time (RAM-safe)."""
    _, batch_paths = load_manifest(manifest_path)
    for p in batch_paths:
        payload = torch.load(p, map_location='cpu', weights_only=False)
        graphs = payload['graphs'] if isinstance(payload, dict) and 'graphs' in payload else payload
        yield p, graphs


def iter_graph_batches_from_paths(batch_paths):
    """Yield graph lists one serialized batch at a time from explicit file paths."""
    for p in batch_paths:
        payload = torch.load(p, map_location='cpu', weights_only=False)
        graphs = payload['graphs'] if isinstance(payload, dict) and 'graphs' in payload else payload
        yield p, graphs


def split_batch_files(batch_paths, val_ratio=0.2, seed=42):
    """Split serialized graph-batch files into train and validation subsets."""
    if val_ratio <= 0.0 or len(batch_paths) <= 1:
        return list(batch_paths), []

    shuffled = list(batch_paths)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1

    val_paths = shuffled[:val_count]
    train_paths = shuffled[val_count:]
    return train_paths, val_paths


def compute_macro_f1(y_true, y_pred):
    """Return Macro F1 score or NaN when it cannot be computed."""
    if len(y_true) == 0:
        return float('nan')
    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        pass # F1 can technically compute with 1 class, but keeping semantics similar
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


def evaluate_from_batch_paths(model, batch_paths, criterion, batch_size, device, val_ids=None):
    """Evaluate loss and AP over serialized graph batches using DataLoaders."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_file, graphs in iter_graph_batches_from_paths(batch_paths):
            if val_ids is not None:
                graphs = [g for g in graphs if getattr(g, 'protein_id', getattr(g, 'pdb_id', getattr(g, 'pocket_id', ''))).split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '') in val_ids]
            if not graphs:
                continue

            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            for pyg_batch in loader:
                pyg_batch = pyg_batch.to(device)
                y = pyg_batch.y.view(-1).long()
                logits = model(pyg_batch)
                loss = criterion(logits, y)

                preds = logits.argmax(dim=1)
                bs = int(y.shape[0])

                total_loss += loss.item() * bs
                total_items += bs
                all_labels.extend(y.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())

            del loader
            del graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_loss = (total_loss / total_items) if total_items > 0 else float('nan')
    f1 = compute_macro_f1(np.array(all_labels), np.array(all_preds))
    return avg_loss, f1, total_items


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.counter = 0

    def step(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def load_split_ids_e3():
    train_ids, val_ids = set(), set()
    if os.path.exists('train_e3.txt'):
        train_ids = set(open('train_e3.txt').read().splitlines())
    if os.path.exists('validation_e3.txt'):
        val_ids = set(open('validation_e3.txt').read().splitlines())
    return train_ids, val_ids

def train_from_manifest(manifest_path, epochs=5, batch_size=4, lr=1e-4,
                        hidden_dim=256, num_heads=4, dropout=0.5,
                        num_classes=5, device=None,
                        early_stopping_patience=None,
                        early_stopping_min_delta=0.0,
                        early_stopping_min_epochs=1,
                        val_manifest_path=None,
                        val_ratio=0.2,
                        split_seed=42,
                        save_best_model_path=None):
    """
    Train directly from batched graph files generated by binding_site_graph.py.
    Processes one graph-batch file at a time to keep memory usage low.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model_class = globals().get('GraphClassifier')
    if model_class is None:
        raise ValueError("GraphClassifier nenalezen v modulu train. Ujisti se, že jsi ho podstrčil (např. train.GraphClassifier = GraphClassifierE3)")
        
    model = model_class(
        node_dim=1280,
        hidden_dim=hidden_dim,
        num_attention_heads=num_heads,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    manifest, all_batch_paths = load_manifest(manifest_path)
    logging.info(f"Loaded manifest with {len(all_batch_paths)} graph-batch files")
    logging.info(f"Manifest expected graphs: {manifest.get('num_graphs', 'unknown')}")

    train_ids, val_ids = load_split_ids_e3()
    filter_by_id = len(train_ids) > 0 and len(val_ids) > 0

    if filter_by_id:
        logging.info("Using explicit splits from structure_clustering (train_e3.txt, validation_e3.txt)")
        train_batch_paths = all_batch_paths
        val_batch_paths = all_batch_paths
    elif val_manifest_path:
        _, val_batch_paths = load_manifest(val_manifest_path)
        train_batch_paths = all_batch_paths
        logging.info(f"Validation manifest loaded with {len(val_batch_paths)} graph-batch files")
    else:
        train_batch_paths, val_batch_paths = split_batch_files(
            all_batch_paths,
            val_ratio=val_ratio,
            seed=split_seed,
        )
        logging.info(
            f"Using internal split from training manifest: "
            f"train_files={len(train_batch_paths)}, val_files={len(val_batch_paths)}"
        )

    use_early_stopping = (
        early_stopping_patience is not None and early_stopping_patience > 0
    )
    early_stopper = None
    if use_early_stopping:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        )
        print(
            "Early stopping enabled: "
            f"patience={early_stopping_patience}, "
            f"min_delta={early_stopping_min_delta}, "
            f"min_epochs={early_stopping_min_epochs}"
        )

    best_val_f1 = -float('inf')
    best_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_items = 0

        all_train_labels = []
        all_train_preds = []

        for batch_file, graphs in iter_graph_batches_from_paths(train_batch_paths):
            if filter_by_id:
                graphs = [g for g in graphs if getattr(g, 'protein_id', getattr(g, 'pdb_id', getattr(g, 'pocket_id', ''))).split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '') in train_ids]
            if not graphs:
                continue

            loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
            for pyg_batch in loader:
                pyg_batch = pyg_batch.to(device)
                y = pyg_batch.y.view(-1).long()

                optimizer.zero_grad(set_to_none=True)
                logits = model(pyg_batch)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                preds = logits.detach().argmax(dim=1)

                bs = int(y.shape[0])
                epoch_loss += loss.item() * bs
                epoch_items += bs
                all_train_labels.extend(y.detach().cpu().numpy().tolist())
                all_train_preds.extend(preds.cpu().numpy().tolist())

            del loader
            del graphs


        avg_loss = (epoch_loss / epoch_items) if epoch_items > 0 else float('nan')
        train_f1 = compute_macro_f1(
            np.array(all_train_labels),
            np.array(all_train_preds),
        )

        val_loss = float('nan')
        val_f1 = float('nan')
        val_items = 0
        if val_batch_paths:
            val_loss, val_f1, val_items = evaluate_from_batch_paths(
                model=model,
                batch_paths=val_batch_paths,
                criterion=criterion,
                batch_size=batch_size,
                device=device,
                val_ids=val_ids if filter_by_id else None
            )

        logging.info(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss: {avg_loss:.6f} - train_macroF1: {train_f1:.6f} - train_samples: {epoch_items} - "
            f"val_loss: {val_loss:.6f} - val_macroF1: {val_f1:.6f} - val_samples: {val_items}"
        )

        if save_best_model_path and not np.isnan(val_f1) and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'val_f1': float(val_f1),
                    'model_state_dict': model.state_dict(),
                },
                save_best_model_path,
            )
            print(
                f"  saved best checkpoint by val_macroF1={val_f1:.6f} "
                f"to: {save_best_model_path}"
            )

        if use_early_stopping and epoch >= early_stopping_min_epochs and epoch_items > 0:
            monitor_metric = val_f1 if not np.isnan(val_f1) else train_f1
            if np.isnan(monitor_metric):
                print("  early-stopping skipped: F1 metric is NaN")
                continue

            # Monitorujeme F1 (cim vyssi, tim lepsi).
            should_stop = early_stopper.step(monitor_metric)
            best_metric = early_stopper.best_score
            wait = early_stopper.counter
            print(
                "  early-stopping status: "
                f"best_F1={best_metric:.6f}, "
                f"wait={wait}/{early_stopping_patience}"
            )
            if should_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if save_best_model_path:
        if best_epoch is None:
            print(
                "Best checkpoint was not saved because validation F1 was NaN "
                "in all epochs."
            )
        else:
            print(
                f"Best checkpoint summary: epoch={best_epoch}, "
                f"val_macroF1={best_val_f1:.6f}, path={save_best_model_path}"
            )

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train GNN directly from graph batches manifest (RAM-safe streaming).'
    )
    parser.add_argument(
        '--graph-manifest',
        default='binding_site_graph_batches/manifest.json',
        help='Path to manifest.json generated by binding_site_graph.py',
    )
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--val-manifest',
        default=None,
        help='Optional manifest.json for validation set (if omitted, uses --val-ratio split).',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation file ratio when --val-manifest is not provided.',
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=42,
        help='Random seed for train/validation split from manifest files.',
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=0,
        help='Enable early stopping with given patience (0 disables).',
    )
    parser.add_argument(
        '--early-stopping-min-delta',
        type=float,
        default=0.0,
        help='Minimum improvement in monitored score to reset patience.',
    )
    parser.add_argument(
        '--early-stopping-min-epochs',
        type=int,
        default=1,
        help='Do not apply early stopping before this epoch.',
    )
    parser.add_argument('--save-model', default='gnn_model.pt')
    parser.add_argument(
        '--save-best-model',
        default='gnn_model_best.pt',
        help='Path to best checkpoint saved by validation F1 (set empty string to disable).',
    )
    args = parser.parse_args()

    if not os.path.exists(args.graph_manifest):
        raise FileNotFoundError(f"Manifest not found: {args.graph_manifest}")

    model = train_from_manifest(
        manifest_path=args.graph_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_classes=args.num_classes,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        val_manifest_path=args.val_manifest,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        save_best_model_path=(args.save_best_model if args.save_best_model else None),
    )

    torch.save(model.state_dict(), args.save_model)
    logging.info(f"Saved model to: {args.save_model}")