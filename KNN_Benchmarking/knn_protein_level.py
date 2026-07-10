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

def load_data_and_pool(embeddings_path, pooling_mode='mean'):
    print(f"Načítám embeddingy z {embeddings_path}...")
    data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    
    X = data['embeddings']
    if hasattr(X, 'numpy'):
        X = X.numpy()
        
    labels = data['labels']
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    protein_ids = data['protein_ids']
    
    # Seskupení kapes podle proteinů
    bags = defaultdict(list)
    bag_labels = {}
    
    for i, pid in enumerate(protein_ids):
        bags[pid].append(X[i])
        bag_labels[pid] = labels[i]
        
    bag_list = []
    for pid, features_list in bags.items():
        features_np = np.stack(features_list)
        
        # Agregace (Pooling)
        if pooling_mode == 'mean':
            pooled_feat = np.mean(features_np, axis=0)
        elif pooling_mode == 'max':
            pooled_feat = np.max(features_np, axis=0)
        else:
            raise ValueError(f"Neznámý pooling: {pooling_mode}")
            
        bag_list.append({
            'protein_id': pid,
            'features': pooled_feat,
            'label': bag_labels[pid]
        })
        
    print(f"Celkem proteinů po {pooling_mode} poolingu: {len(bag_list)}")
    return bag_list

def main():
    parser = argparse.ArgumentParser(description="KNN Benchmark (Protein Level)")
    parser.add_argument('--embeddings-path', default='../p2rank_egnn_embeddings.pt')
    parser.add_argument('--pooling', choices=['mean', 'max'], default='mean', 
                        help='Jak agregovat kapsy do jednoho proteinového vektoru')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(args.embeddings_path))
    
    if not os.path.exists(args.embeddings_path):
        print(f"Soubor {args.embeddings_path} nenalezen!")
        return
        
    bags = load_data_and_pool(args.embeddings_path, pooling_mode=args.pooling)
    train_ids, val_ids, test_ids = load_split_ids(base_dir)
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_fold = []
    
    # Rozdělení na train a test
    if train_ids and test_ids and val_ids:
        print("Používám existující splity (train/val/test_mil.txt) pro přesnou PredefinedSplit validaci...")
        for b in bags:
            pid = b['protein_id']
            # Chceme evaluovat KNN hlavně na Test setu
            if pid in test_ids:
                X_test.append(b['features'])
                y_test.append(b['label'])
            elif pid in val_ids:
                X_train.append(b['features'])
                y_train.append(b['label'])
                test_fold.append(0)  # 0 znamená validaci v GridSearch
            elif pid in train_ids:
                X_train.append(b['features'])
                y_train.append(b['label'])
                test_fold.append(-1) # -1 znamená čistý trénink v GridSearch
        cv = PredefinedSplit(test_fold)
    else:
        print("Splity nenalezeny, padám na náhodný split 80/20 (5-fold CV)...")
        np.random.seed(42)
        np.random.shuffle(bags)
        split_idx = int(len(bags) * 0.8)
        for b in bags[:split_idx]:
            X_train.append(b['features'])
            y_train.append(b['label'])
        for b in bags[split_idx:]:
            X_test.append(b['features'])
            y_test.append(b['label'])
        cv = 5
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nTrain set: {len(X_train)} proteinů")
    print(f"Test set:  {len(X_test)} proteinů")
    
    print("\nSpouštím KNN Hyperparameter Optimization (GridSearchCV)...")
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'cosine', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print(f"\nNejlepší nalezené parametry pro KNN (protein-level, {args.pooling} pooling):")
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
