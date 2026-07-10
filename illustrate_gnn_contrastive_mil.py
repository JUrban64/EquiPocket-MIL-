import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import torch
import argparse
import os

def load_real_mil_data(embeddings_path):
    print(f"Loading real data from {embeddings_path}...")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"File {embeddings_path} not found.")
        
    try:
        data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    except Exception as e:
        data = torch.load(embeddings_path, map_location='cpu')
    
    if 'embeddings' not in data or 'labels' not in data:
        raise ValueError(f"File {embeddings_path} missing 'embeddings' or 'labels' keys.")
        
    embeddings = data['embeddings']
    labels = data['labels']
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
        
    print(f"Successfully loaded {embeddings.shape[0]} graphs/patches with dimension {embeddings.shape[1]}.")
    return embeddings, labels

def plot_real_embeddings(embeddings_path, save_path="gnn_embeddings_umap.png"):
    embeddings, labels = load_real_mil_data(embeddings_path)
    
    print("Applying UMAP to reduce to 2D (this might take a few moments)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    umap_result = reducer.fit_transform(embeddings)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    print(f"Found unique classes (labels): {unique_labels}")
    
    # Define color palette for multiple classes
    palette_list = sns.color_palette("husl", len(unique_labels))
    palette = {lbl: col for lbl, col in zip(unique_labels, palette_list)}
    
    # Sort labels for consistent plotting
    sorted_labels = sorted(unique_labels)
    
    for lbl in sorted_labels:
        # Tady už ke všem třídám (0, 1, 2, 3, 4) přistupujeme rovnocenně, 
        # protože všechny představují validní vazebná místa, nikoliv pozadí.
        sns.scatterplot(
            x=umap_result[labels==lbl, 0], y=umap_result[labels==lbl, 1], 
            color=palette[lbl], alpha=0.8, s=50, edgecolor='black', linewidth=0.5, 
            label=f'Class {lbl}'
        )
        
    title_suffix = "Ground Truth Data" if "gt" in embeddings_path.lower() else "P2Rank Pockets (MIL)"
    plt.title(f"UMAP Projection of EGNN Embeddings ({title_suffix})\nContrastive Encoder Class Separation", fontsize=14, pad=15)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    
    # Legenda umístěna chytře vedle grafu, aby nepřekrývala data
    plt.legend(title="Binding Site Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization successfully saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot UMAP of real GNN pocket embeddings.")
    parser.add_argument('--embeddings-path', type=str, default='gt_egnn_embeddings.pt', help='Path to the .pt file with embeddings.')
    parser.add_argument('--save-path', type=str, default='gnn_embeddings_umap.png', help='Path to save the generated image.')
    args = parser.parse_args()
    
    try:
        plot_real_embeddings(args.embeddings_path, args.save_path)
    except Exception as e:
        print(f"Error during processing: {e}")
