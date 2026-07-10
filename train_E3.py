import sys
import os
import argparse
import torch


import train
from model_E3 import GraphClassifierE3

# Mimořádně Pythonic způsob - jednoduše podsunieme našemu existujícímu kódu z train.py 
# novou třídu E3 modelu. Všechno ostatní poběží stejně.
train.GraphClassifier = GraphClassifierE3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train E(3)-Equivariant GNN directly from graph batches manifest.'
    )
    # Odkaž se na vygenerovaná data v SQBCP složce:
    parser.add_argument(
        '--graph-manifest',
        default='gt_graph_batches/manifest.json',
    )
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularization weight decay.')
    parser.add_argument('--no-scheduler', action='store_false', dest='use_scheduler', help='Disable learning rate scheduler.')
    parser.set_defaults(use_scheduler=True)
    parser.add_argument('--scheduler-patience', type=int, default=10, help='Patience for learning rate scheduler.')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, help='Multiplicative factor of learning rate decay.')
    parser.add_argument('--ce-weight', type=float, default=0.0, help='Weight of cross-entropy loss.')
    parser.add_argument('--contrastive-weight', type=float, default=1.0, help='Weight of supervised contrastive loss.')
    parser.add_argument('--contrastive-temp', type=float, default=0.07, help='Temperature parameter for contrastive learning.')
    
    parser.add_argument('--hidden-dim', type=int, default=128) 
    
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--val-manifest',
        default=None,
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
    )
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--early-stopping-patience', type=int, default=0)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0)
    parser.add_argument('--early-stopping-min-epochs', type=int, default=1)
    
    # Výchozí jména modelů pro E3 pipeline
    parser.add_argument('--save-model', default='gnn_model_e3.pt')
    parser.add_argument('--save-best-model', default='gnn_model_best_e3.pt')
    args = parser.parse_args()

    if not os.path.exists(args.graph_manifest):
        raise FileNotFoundError(f"Manifest not found: {args.graph_manifest}")

    model = train.train_from_manifest(
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
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        ce_weight=args.ce_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temp=args.contrastive_temp,
    )

    torch.save(model.state_dict(), args.save_model)
    print(f"Saved model to: {args.save_model}")
