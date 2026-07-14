import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataset_esm import load_data_from_pyg_batches, load_split_ids
from model_mil import AttentionMIL_ESM

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # 1. Load data
    bags = load_data_from_pyg_batches(args.manifest_path, mode=args.mode)
    
    # 2. Split data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_ids, val_ids, test_ids = load_split_ids(base_dir)
    
    train_bags, val_bags, test_bags = [], [], []
    
    if train_ids and test_ids:
        print("Using splits from text files...")
        for b in bags:
            pid = b['protein_id']
            if pid in train_ids:
                train_bags.append(b)
            elif pid in val_ids:
                val_bags.append(b)
            elif pid in test_ids:
                test_bags.append(b)
            else:
                train_bags.append(b)
    else:
        print("Split files not found. Using random 80/10/10 split...")
        np.random.seed(42)
        np.random.shuffle(bags)
        n = len(bags)
        train_bags = bags[:int(n*0.8)]
        val_bags = bags[int(n*0.8):int(n*0.9)]
        test_bags = bags[int(n*0.9):]
        
    print(f"Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    
    if len(train_bags) == 0:
        return
        
    # 3. Model setup
    all_labels = set([b['label'].item() for b in bags])
    num_classes = max(all_labels) + 1
    
    # ESM dim is 1280
    model = AttentionMIL_ESM(
        in_features=1280,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        num_heads=args.num_heads,
        attention_temp=args.attention_temp,
        gated_attention=args.gated_attention
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopper = EarlyStopping(patience=args.patience)
    
    best_val_loss = float('inf')
    
    print("\n--- Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        np.random.shuffle(train_bags)
        
        optimizer.zero_grad()
        batch_count = 0
        
        for i, bag in enumerate(train_bags):
            features = bag['features'].to(device)
            label = bag['label'].to(device)
            
            logits, A = model(features)
            loss = criterion(logits, label)
            
            loss_scaled = loss / args.batch_size
            loss_scaled.backward()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            if batch_count == args.batch_size or i == len(train_bags) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0
                
        avg_train_loss = total_train_loss / len(train_bags)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for bag in val_bags:
                features = bag['features'].to(device)
                label = bag['label'].to(device)
                logits, _ = model(features)
                val_loss = criterion(logits, label)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_bags) if len(val_bags) > 0 else avg_train_loss
        
        if early_stopper(avg_val_loss):
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_path)
            
        if (epoch + 1) % 5 == 0 or epoch == 0 or early_stopper.early_stop:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Best Val: {best_val_loss:.4f} | Patience: {early_stopper.counter}/{args.patience}")
            
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    print(f"\nLoading best model from {args.model_path} for evaluation...")
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
    model.eval()
    
    # 4. Evaluation
    eval_bags = test_bags if args.evaluate_test and len(test_bags) > 0 else val_bags
    eval_name = "Test" if args.evaluate_test and len(test_bags) > 0 else "Validation"
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for bag in eval_bags:
            features = bag['features'].to(device)
            logits, A = model(features)
            pred = torch.argmax(logits, dim=1).item()
            
            y_true.append(bag['label'].item())
            y_pred.append(pred)
            
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n--- Results on {eval_name} Set ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-path', default='../p2rank_graph_batches/manifest.json')
    parser.add_argument('--mode', choices=['pockets', 'residues'], default='pockets', 
                        help='Mód dat: pockets (1 vektor per kapsa) nebo residues (1 vektor per aminokyselina)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--attention-temp', type=float, default=1.0)
    parser.add_argument('--gated-attention', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--model-path', default='best_esm_mil.pt')
    
    args = parser.parse_args()
    train_and_evaluate(args)
