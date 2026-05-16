import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from collections import defaultdict

# Nastavení logování
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("mil_training.log"),
        logging.StreamHandler()
    ]
)

class AttentionMIL(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=6):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: [num_pockets, in_features]
        A = self.attention(x)  # [num_pockets, 1]
        A = torch.softmax(A, dim=0)  # Softmax přes všechny kapsy
        
        # Seskupení kapes do jednoho vektoru pro celý protein (vážený součet)
        Z = torch.mm(A.t(), x)  # [1, num_pockets] x [num_pockets, in_features] = [1, in_features]
        
        # Klasifikace proteinu
        logits = self.classifier(Z)  # [1, num_classes]
        return logits, A

def load_data_and_group(embeddings_path):
    logging.info(f"Loading embeddings from {embeddings_path}...")
    data = torch.load(embeddings_path, weights_only=False)
    
    X = data['embeddings']
    labels = data['labels']
    protein_ids = data['protein_ids']
    pocket_ids = data['pocket_ids']
    
    # Seskupení do "bagů"
    bags = defaultdict(list)
    bag_labels = {}
    bag_pockets = defaultdict(list)
    
    for i, pid in enumerate(protein_ids):
        bags[pid].append(X[i])
        bag_labels[pid] = labels[i]  # Label je stejný pro všechny kapsy z jednoho proteinu
        bag_pockets[pid].append(pocket_ids[i])
        
    bag_list = []
    for pid in bags.keys():
        bag_list.append({
            'protein_id': pid,
            'features': torch.FloatTensor(np.stack(bags[pid])),
            'label': torch.LongTensor([bag_labels[pid]]),
            'pocket_ids': bag_pockets[pid]
        })
        
    logging.info(f"Total proteins (bags): {len(bag_list)}")
    return bag_list

def load_split_ids():
    train_ids, val_ids, test_ids = set(), set(), set()
    if os.path.exists('train_mil.txt'):
        train_ids = set(open('train_mil.txt').read().splitlines())
    if os.path.exists('validation_mil.txt'):
        val_ids = set(open('validation_mil.txt').read().splitlines())
    if os.path.exists('test_mil.txt'):
        test_ids = set(open('test_mil.txt').read().splitlines())
    return train_ids, val_ids, test_ids

def train_and_evaluate(bags, epochs=50, lr=1e-3):
    train_ids, val_ids, test_ids = load_split_ids()
    
    train_bags, val_bags, test_bags = [], [], []
    
    if train_ids and test_ids:
        logging.info("Using splits from structure_clustering...")
        for b in bags:
            pid = b['protein_id']
            if pid in train_ids:
                train_bags.append(b)
            elif pid in val_ids:
                val_bags.append(b)
            elif pid in test_ids:
                test_bags.append(b)
            else:
                # Fallback, když není ve splitech
                train_bags.append(b)
    else:
        logging.info("Split files not found. Using random 80/20 split...")
        np.random.seed(42)
        np.random.shuffle(bags)
        split_idx = int(len(bags) * 0.8)
        train_bags = bags[:split_idx]
        test_bags = bags[split_idx:]

    logging.info(f"Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    
    if len(train_bags) == 0:
        logging.error("Error: No training data.")
        return
        
    in_features = train_bags[0]['features'].shape[1]
    
    # Najdeme všechny unikátní labely abychom zjistili num_classes
    all_labels = set()
    for b in bags:
        all_labels.add(b['label'].item())
    num_classes = max(all_labels) + 1
    
    model = AttentionMIL(in_features=in_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    logging.info("\\n--- Trénink MIL sítě ---")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        np.random.shuffle(train_bags)
        
        for bag in train_bags:
            optimizer.zero_grad()
            logits, A = model(bag['features'])
            loss = criterion(logits, bag['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss/len(train_bags)
        
        # Uložení nejlepšího modelu
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "mil_model_best.pt")
            
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Best Loss: {best_loss:.4f}")
            
    # Načtení nejlepšího modelu pro vyhodnocení
    logging.info("\\nNačítám nejlepší model pro vyhodnocení...")
    if os.path.exists("mil_model_best.pt"):
        model.load_state_dict(torch.load("mil_model_best.pt"))
        
    # Vyhodnocení
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    
    eval_bags = test_bags if len(test_bags) > 0 else train_bags
    
    logging.info("\\n--- Výsledky na Testovací sadě ---")
    with torch.no_grad():
        for bag in eval_bags:
            logits, A = model(bag['features'])
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred = np.argmax(probs)
            
            y_true.append(bag['label'].item())
            y_pred.append(pred)
            y_probs.append(probs)
            
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test Macro F1: {f1:.4f}")
    
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        logging.info(f"Test ROC AUC:  {auc:.4f}")
    except ValueError:
        pass # Pokud chybí nějaká třída v test setu
        
    logging.info("\\nClassification Report:")
    logging.info("\\n" + classification_report(y_true, y_pred, zero_division=0))
    
    # Ukázka Attention vah
    logging.info("\\n--- Ukázka nalezených kapes (Attention) ---")
    logging.info("Model se sám naučil vybrat nejdůležitější p2rank kapsu z proteinu:")
    show_limit = 3
    shown = 0
    with torch.no_grad():
        for bag in eval_bags:
            if shown >= show_limit: break
            logits, A = model(bag['features'])
            A = A.squeeze().numpy()
            if A.ndim == 0:
                A = [A.item()]
            
            pockets = bag['pocket_ids']
            best_idx = np.argmax(A)
            
            output_lines = [f"Protein: {bag['protein_id']} (True Class: {bag['label'].item()})"]
            for i, p in enumerate(pockets):
                marker = "<-- [VYBRÁNO MODELEM]" if i == best_idx else ""
                output_lines.append(f"  {p}: váha {A[i]:.4f} {marker}")
            logging.info("\\n" + "\\n".join(output_lines))
            shown += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention MIL classifier on EGNN embeddings")
    parser.add_argument('--embeddings-path', default='p2rank_egnn_embeddings.pt')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    if not os.path.exists(args.embeddings_path):
        print(f"File {args.embeddings_path} not found.")
        exit(1)
        
    bags = load_data_and_group(args.embeddings_path)
    train_and_evaluate(bags, epochs=args.epochs)
