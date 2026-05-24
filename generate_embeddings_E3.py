import sys
import os
import argparse
import torch
import numpy as np



from torch_geometric.loader import DataLoader
from model_E3 import GraphClassifierE3
import train

def generate_embeddings(
    manifest_path,
    model_path,
    output_path,
    batch_size=4,
    hidden_dim=64,
    num_heads=4,
    num_classes=5,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Initializing model (hidden_dim={hidden_dim}, num_heads={num_heads})...")
    model = GraphClassifierE3(
        node_dim=1280,
        hidden_dim=hidden_dim,
        num_attention_heads=num_heads,
        num_classes=num_classes,
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading checkpoint from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()

    manifest, all_batch_paths = train.load_manifest(manifest_path)
    print(f"Loaded manifest with {len(all_batch_paths)} graph-batch files")

    all_embeddings = []
    all_labels = []
    all_pdb_ids = []
    all_is_correct = []
    all_pocket_ids = []

    print("Generating embeddings...")
    with torch.no_grad():
        for batch_file, graphs in train.iter_graph_batches_from_paths(all_batch_paths):
            if not graphs:
                continue

            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            for pyg_batch in loader:
                pyg_batch = pyg_batch.to(device)
                
                # Získání reprezentace grafu (před klasifikací)
                embeddings = model.get_embedding(pyg_batch)
                
                # Převedeme do numpy pro snazší ukládání
                emb_np = embeddings.cpu().numpy()
                y_np = pyg_batch.y.view(-1).cpu().numpy()
                
                # Uložení výsledků pro tento batch
                all_embeddings.append(emb_np)
                all_labels.append(y_np)
                
                # PyG Dataloader automaticky spojí string atributy do seznamu
                if hasattr(pyg_batch, 'pdb_id'):
                    all_pdb_ids.extend(pyg_batch.pdb_id)
                elif hasattr(pyg_batch, 'protein_id'):
                    all_pdb_ids.extend(pyg_batch.protein_id)
                else:
                    all_pdb_ids.extend([f"unknown_{i}" for i in range(len(y_np))])

                if hasattr(pyg_batch, 'pocket_id'):
                    all_pocket_ids.extend(pyg_batch.pocket_id)

            del loader
            del graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Sloučení všech batched polí do jednoho velkého numpy pole
    if len(all_embeddings) == 0:
        print("No embeddings generated! Check your manifest and data.")
        return

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Total embeddings generated: {final_embeddings.shape}")
    
    results = {
        'embeddings': final_embeddings,
        'labels': final_labels,
        'pdb_ids': all_pdb_ids
    }
    
    if len(all_pocket_ids) > 0:
        results['pocket_ids'] = all_pocket_ids
        # Sestavime protein_ids z pocket_ids pro snadné MIL groupování
        protein_ids = [pid.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '') for pid in all_pocket_ids]
        results['protein_ids'] = protein_ids
    
    torch.save(results, output_path)
    print(f"Embeddings successfully saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate EGNN graph embeddings from trained model.'
    )
    parser.add_argument(
        '--graph-manifest',
        default='gt_graph_batches/manifest.json',
    )
    parser.add_argument(
        '--model-path',
        default='gnn_model_e3.pt',
        help='Path to the trained E3 model checkpoint'
    )
    parser.add_argument(
        '--output-path',
        default='egnn_embeddings.pt',
        help='Where to save the resulting dictionary of embeddings'
    )
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--device', default=None)
    
    args = parser.parse_args()
    
    generate_embeddings(
        manifest_path=args.graph_manifest,
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        device=args.device
    )
