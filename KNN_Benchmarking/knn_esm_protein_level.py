import torch
import numpy as np
import os
import json
import argparse
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

def load_split_ids(base_dir):
    train_ids, val_ids, test_ids = set(), set(), set()
    train_path = os.path.join(base_dir, 'train_mil.txt')
    val_path = os.path.join(base_dir, 'validation_mil.txt')
    test_path = os.path.join(base_dir, 'test_mil.txt')
    
    if os.path.exists(train_path):
        train_ids = set(open(train_path).read().splitlines())
    if os.path.exists(val_path):
        val_ids = set(open(val_path).read().splitlines())
    if os.path.exists(test_path):
        test_ids = set(open(test_path).read().splitlines())
    return train_ids, val_ids, test_ids

def load_esm_proteins_and_pool(manifest_path, pooling_mode='mean'):
    print(f"Načítám ESM embeddingy z PyG grafů podle {manifest_path}...")
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    # Seskupení kapes podle proteinů
    protein_pockets = defaultdict(list)
    protein_labels = {}
    
    for batch_file in tqdm(manifest['batch_files'], desc="Načítání grafů"):
        full_path = os.path.join(base_dir, batch_file)
        if not os.path.exists(full_path):
            continue
            
        data_batch = torch.load(full_path, map_location='cpu', weights_only=False)
        graphs = data_batch.get('graphs', [])
        
        for g in graphs:
            if g.x is None or g.x.shape[0] == 0:
                continue
                
            # 1. Nejprve zprůměrujeme ESM embeddingy residuí, abychom získali vektor kapsy
            esm_pocket = g.x.mean(dim=0).numpy()
            
            # Label
            label = g.y.item() if hasattr(g.y, 'item') else int(g.y[0])
            
            # Protein ID
            pocket_id = getattr(g, 'pocket_id', '')
            if pocket_id:
                protein_id = pocket_id.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            else:
                pdb_id = getattr(g, 'pdb_id', '')
                protein_id = pdb_id.replace('.pdb', '')
                
            protein_pockets[protein_id].append(esm_pocket)
            protein_labels[protein_id] = label
            
    # 2. Poolování kapes na úroveň proteinu
    protein_list = []
    for pid, pocket_feats in protein_pockets.items():
        feats_np = np.stack(pocket_feats)
        
        if pooling_mode == 'mean':
            pooled_feat = np.mean(feats_np, axis=0)
        elif pooling_mode == 'max':
            pooled_feat = np.max(feats_np, axis=0)
        else:
            raise ValueError(f"Neznámý pooling: {pooling_mode}")
            
        protein_list.append({
            'protein_id': pid,
            'features': pooled_feat,
            'label': protein_labels[pid]
        })
        
    print(f"Celkem proteinů (s ESM {pooling_mode} poolingem): {len(protein_list)}")
    return protein_list

def main():
    parser = argparse.ArgumentParser(description="KNN Benchmark (Pure ESM Protein Level)")
    parser.add_argument('--manifest-path', default='../p2rank_graph_batches/manifest.json')
    parser.add_argument('--pooling', choices=['mean', 'max'], default='mean', 
                        help='Jak agregovat ESM kapsy do jednoho proteinového vektoru')
    args = parser.parse_args()
    
    # Split text files are assumed to be in the parent dir of KNN_benchmarking
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.exists(args.manifest_path):
        print(f"Soubor {args.manifest_path} nenalezen!")
        return
        
    proteins = load_esm_proteins_and_pool(args.manifest_path, pooling_mode=args.pooling)
    train_ids, val_ids, test_ids = load_split_ids(base_dir)
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_fold = []
    
    # Rozdělení na train a test podle protein_id
    if train_ids and test_ids and val_ids:
        print("Používám existující splity (train/val/test_mil.txt) pro přesnou PredefinedSplit validaci...")
        for p in proteins:
            pid = p['protein_id']
            if pid in test_ids:
                X_test.append(p['features'])
                y_test.append(p['label'])
            elif pid in val_ids:
                X_train.append(p['features'])
                y_train.append(p['label'])
                test_fold.append(0)  # Validace
            elif pid in train_ids:
                X_train.append(p['features'])
                y_train.append(p['label'])
                test_fold.append(-1) # Trénink
        cv = PredefinedSplit(test_fold)
    else:
        print("Splity nenalezeny, padám na náhodný split po proteinech (5-fold CV)...")
        np.random.seed(42)
        np.random.shuffle(proteins)
        split_idx = int(len(proteins) * 0.8)
        for p in proteins[:split_idx]:
            X_train.append(p['features'])
            y_train.append(p['label'])
        for p in proteins[split_idx:]:
            X_test.append(p['features'])
            y_test.append(p['label'])
        cv = 5
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nTrain set: {len(X_train)} proteinů")
    print(f"Test set:  {len(X_test)} proteinů")
    
    print("\nSpouštím KNN Hyperparameter Optimization (GridSearchCV)...")
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'cosine']
    }
    
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print(f"\nNejlepší nalezené parametry pro KNN (pure ESM protein-level, {args.pooling} pooling):")
    print(clf.best_params_)
    
    print("\nEvaluace na Test setu:")
    best_knn = clf.best_estimator_
    y_pred = best_knn.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == '__main__':
    main()
