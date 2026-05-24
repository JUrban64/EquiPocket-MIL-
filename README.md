# PMCP: Protein Pocket MIL Classification Pipeline

An end-to-end architecture for protein binding site extraction, embedding generation via E(3)-Equivariant Graph Neural Networks (EGNN), and subsequent cofactor classification using Attention-based Multiple Instance Learning (MIL).

## Architecture

1. **Graph Builder:** Extracts 3D coordinates and ESM sequence embeddings from protein binding sites to construct PyTorch Geometric graphs.
2. **EGNN Encoder:** An E(3)-equivariant network trained with a Joint Loss (Cross-Entropy + Supervised Contrastive Learning) to act as a robust 3D structural pocket encoder. It produces rotation-invariant representations.
3. **Attention-based MIL:** A Multiple Instance Learning classifier that receives all P2Rank-predicted pockets of a single protein (a "bag"). Using the Attention mechanism, it automatically identifies the most biologically relevant pocket, assigns it a high weight, and classifies the required cofactor (e.g., ATP, NAD, FAD).

---

## Pipeline Usage

### 1. Extract Ground Truth & Build Graphs
Prepare the crystallographic ground truth (GT) graphs for EGNN training:
```bash
python build_p2rank_dataset.py
python binding_site_graph.py \
    --input-json binding_sites_by_protein.json \
    --output-dir gt_graph_batches
```

### 2. Train EGNN Encoder
Train the EGNN model to recognize 3D pocket geometries and generate high-quality embeddings.
```bash
python train_E3.py \
    --epochs 350 \
    --batch-size 8 \
    --hidden-dim 64 \
    --contrastive-weight 0.5 \
    --num-classes 5
```

### 3. Prepare P2Rank Data
Build graphs from unlabelled P2Rank pocket predictions:
```bash
python binding_site_graph.py \
    --input-json p2rank_pockets_dataset.json \
    --output-dir p2rank_graph_batches
```

### 4. Generate Embeddings
Extract embeddings for all P2Rank pockets using the trained EGNN encoder:
```bash
python generate_embeddings_E3.py \
    --model-path gnn_model_best_e3.pt \
    --graph-manifest p2rank_graph_batches/manifest.json \
    --output-path p2rank_egnn_embeddings.pt
```

### 5. Train MIL Classifier
Train the MIL network to classify the correct cofactor from the bag of pocket embeddings:
```bash
python train_mil_classifier.py \
    --embeddings-path p2rank_egnn_embeddings.pt \
    --epochs 150
```
*(Training logs and attention weights for selected pockets are saved automatically).*
