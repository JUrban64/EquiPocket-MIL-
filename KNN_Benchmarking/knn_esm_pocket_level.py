import torch
import numpy as np
import os
import json
import argparse
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

def load_esm_pockets(manifest_path):
    print(f"Načítám ESM embeddingy přímo z PyG grafů podle {manifest_path}...")
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    pocket_list = []
    
    for batch_file in tqdm(manifest['batch_files'], desc="Načítání grafů"):
        full_path = os.path.join(base_dir, batch_file)
        if not os.path.exists(full_path):
            continue
            
        data_batch = torch.load(full_path, map_location='cpu', weights_only=False)
        graphs = data_batch.get('graphs', [])
        
        for g in graphs:
            if g.x is None or g.x.shape[0] == 0:
                continue
                
            # Hodnoty g.x představují ESM embeddingy uzlů. 
            # Zprůměrujeme je přes všechny aminokyseliny, abychom dostali jeden 1280-dim vektor za kapsu.
            esm_pocket = g.x.mean(dim=0).numpy()
            
            # Label
            label = g.y.item() if hasattr(g.y, 'item') else int(g.y[0])
            
            # Protein ID (z pocket_id nebo pdb_id)
            pocket_id = getattr(g, 'pocket_id', '')
            if pocket_id:
                protein_id = pocket_id.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            else:
                pdb_id = getattr(g, 'pdb_id', '')
                protein_id = pdb_id.replace('.pdb', '')
                
            pocket_list.append({
                'protein_id': protein_id,
                'features': esm_pocket,
                'label': label
            })
            
    print(f"Celkem kapes (s ESM): {len(pocket_list)}")
    return pocket_list

def main():
    parser = argparse.ArgumentParser(description="KNN Benchmark (Pure ESM Pocket Level)")
    parser.add_argument('--manifest-path', default='../p2rank_graph_batches/manifest.json')
    args = parser.parse_args()
    
    # Split text files are assumed to be in the parent dir of KNN_benchmarking
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.exists(args.manifest_path):
        print(f"Soubor {args.manifest_path} nenalezen!")
        return
        
    pockets = load_esm_pockets(args.manifest_path)
    train_ids, val_ids, test_ids = load_split_ids(base_dir)
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_fold = []
    
    # Rozdělení na train a test podle protein_id
    if train_ids and test_ids and val_ids:
        print("Používám existující splity (train/val/test_mil.txt) pro přesnou PredefinedSplit validaci...")
        for p in pockets:
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
        unique_pids = list(set([p['protein_id'] for p in pockets]))
        np.random.seed(42)
        np.random.shuffle(unique_pids)
        split_idx = int(len(unique_pids) * 0.8)
        train_pids = set(unique_pids[:split_idx])
        test_pids = set(unique_pids[split_idx:])
        
        for p in pockets:
            if p['protein_id'] in train_pids:
                X_train.append(p['features'])
                y_train.append(p['label'])
            elif p['protein_id'] in test_pids:
                X_test.append(p['features'])
                y_test.append(p['label'])
        cv = 5
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nTrain set: {len(X_train)} kapes")
    print(f"Test set:  {len(X_test)} kapes")
    
    print("\nSpouštím KNN Hyperparameter Optimization (GridSearchCV)...")
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'cosine']
    }
    
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print(f"\nNejlepší nalezené parametry pro KNN (pure ESM pocket-level):")
    print(clf.best_params_)
    
    print("\nEvaluace na Test setu (po kapsách):")
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
