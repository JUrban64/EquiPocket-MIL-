import torch
import numpy as np
import os
import argparse
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

def load_data_pockets(embeddings_path):
    print(f"Načítám embeddingy z {embeddings_path}...")
    data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    
    X = data['embeddings']
    if hasattr(X, 'numpy'):
        X = X.numpy()
        
    labels = data['labels']
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    protein_ids = data['protein_ids']
    
    pocket_list = []
    for i in range(len(X)):
        pocket_list.append({
            'protein_id': protein_ids[i],
            'features': X[i],
            'label': labels[i]
        })
        
    print(f"Celkem kapes: {len(pocket_list)}")
    return pocket_list

def main():
    parser = argparse.ArgumentParser(description="KNN Benchmark (Pocket Level)")
    parser.add_argument('--embeddings-path', default='../p2rank_egnn_embeddings.pt')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(args.embeddings_path))
    
    if not os.path.exists(args.embeddings_path):
        print(f"Soubor {args.embeddings_path} nenalezen!")
        return
        
    pockets = load_data_pockets(args.embeddings_path)
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
                test_fold.append(0)  # 0 znamená, že se vzorek použije pro validaci v GridSearch
            elif pid in train_ids:
                X_train.append(p['features'])
                y_train.append(p['label'])
                test_fold.append(-1) # -1 znamená čistý trénink v GridSearch
        cv = PredefinedSplit(test_fold)
    else:
        print("Splity nenalezeny, padám na náhodný split po proteinech (5-fold CV)...")
        # Musíme splitovat přes unique protein_ids
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
        cv = 5 # default 5-fold CV
            
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
    
    print(f"\nNejlepší nalezené parametry pro KNN (pocket-level):")
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
