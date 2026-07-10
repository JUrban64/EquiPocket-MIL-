import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, average_precision_score
from sklearn.preprocessing import label_binarize
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

class EarlyStopping:
    """
    Early stopping pomocná třída pro sledování validační ztráty.
    """
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
            return True  # Model se zlepšil, uložit
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Bez zlepšení

class AttentionMIL(nn.Module):
    """
    Attention-based Multi-Instance Learning model obohacený o Self-Attention (Transformer Encoder),
    s podporou pro Multi-Head Attention, Gated Attention a teplotním škálováním.
    """
    def __init__(self, in_features, hidden_dim=128, num_classes=5, dropout=0.3, num_heads=1, 
                 attention_temp=1.0, gated_attention=False, use_self_attention=False, self_attn_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_temp = attention_temp
        self.gated_attention = gated_attention
        self.use_self_attention = use_self_attention
        
        # --- PŘIDÁNA SELF-ATTENTION (TRANSFORMER) VRSTVA ---
        if self.use_self_attention:
            # Projekce na hidden_dim zajišťuje dělitelnost počtem hlav (např. 128 je dělitelné 4)
            self.input_proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            )
            # Standardní Transformer Encoder blok (kapsy spolu komunikují)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self_attn_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
            mil_in_features = hidden_dim
        else:
            mil_in_features = in_features
        
        if gated_attention:
            # Gated Attention podle Ilse et al. (2018) s LeakyReLU a LayerNorm
            self.attention_V = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            )
            self.attention_U = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.Sigmoid()
            )
            self.attention_w = nn.Linear(hidden_dim, num_heads)
        else:
            # Klasická Attention s LeakyReLU a LayerNorm stabilizací
            self.attention = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, num_heads)  
            )
        
        # Každá hlava vygeneruje svůj vlastní embedding
        classifier_input_dim = mil_in_features * num_heads
        
        # Klasifikátor zpracovává sloučenou vícehlavou reprezentaci celého proteinu
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: [num_pockets, in_features]
        
        # --- APLIKACE SELF-ATTENTION (KOMUNIKACE MEZI KAPSAMI) ---
        if self.use_self_attention:
            x = self.input_proj(x)              # [num_pockets, hidden_dim]
            x_3d = x.unsqueeze(0)               # Změna na 3D: [1, num_pockets, hidden_dim]
            x_trans = self.self_attention(x_3d) # Kontextualizace: [1, num_pockets, hidden_dim]
            x = x_trans.squeeze(0)              # Zpět na 2D: [num_pockets, hidden_dim]
        
        if self.gated_attention:
            V_out = self.attention_V(x)          # [num_pockets, hidden_dim]
            U_out = self.attention_U(x)          # [num_pockets, hidden_dim]
            gated_out = V_out * U_out             # [num_pockets, hidden_dim]
            A_raw = self.attention_w(gated_out)  # [num_pockets, num_heads]
        else:
            A_raw = self.attention(x)            # [num_pockets, num_heads]
        
        # Aplikace teplotního škálování před softmaxem
        A_scaled = A_raw / self.attention_temp
        
        # Softmax přes dimenzi 0 (kapsy) pro každou hlavu zvlášť
        A = torch.softmax(A_scaled, dim=0)  # [num_pockets, num_heads]
        
        # Seskupení kapes pro každou hlavu zvlášť
        Z_heads = torch.mm(A.t(), x)        # [num_heads, mil_in_features]
        
        # Zploštění všech hlav do jednoho vektoru: [1, num_heads * mil_in_features]
        Z = Z_heads.view(1, -1)
        
        # Klasifikace celého proteinu
        logits = self.classifier(Z)  # [1, num_classes]
        return logits, A

def load_data_and_group(embeddings_path):
    logging.info(f"Loading embeddings from {embeddings_path}...")
    data = torch.load(embeddings_path, weights_only=False)
    
    X = data['embeddings']
    labels = data['labels']
    protein_ids = data['protein_ids']
    pocket_ids = data['pocket_ids']
    
    bags = defaultdict(list)
    bag_labels = {}
    bag_pockets = defaultdict(list)
    
    for i, pid in enumerate(protein_ids):
        bags[pid].append(X[i])
        bag_labels[pid] = labels[i]
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

def train_and_evaluate(bags, epochs=100, lr=1e-3, batch_size=64, hidden_dim=128, dropout=0.3, 
                       weight_decay=1e-4, patience=15, num_heads=1, attention_temp=1.0, 
                       model_path="mil_model_best.pt", evaluate_test=False,
                       gated_attention=False, balance_classes=False,
                       use_self_attention=False, self_attn_heads=4):
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    logging.info(f"Using device: {device}")
    
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
                train_bags.append(b)
    else:
        logging.info("Split files not found. Using random 80/20 split...")
        np.random.seed(42)
        np.random.shuffle(bags)
        split_idx = int(len(bags) * 0.8)
        train_bags = bags[:split_idx]
        test_bags = bags[split_idx:]
        
    if len(val_bags) == 0:
        np.random.seed(42)
        np.random.shuffle(train_bags)
        val_idx = int(len(train_bags) * 0.1)
        val_bags = train_bags[:val_idx]
        train_bags = train_bags[val_idx:]

    logging.info(f"Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    
    if len(train_bags) == 0:
        logging.error("Error: No training data.")
        return
        
    # --- Sanity Check podobnosti embeddingů ---
    for b in train_bags:
        if b['features'].shape[0] > 1:
            cos = torch.nn.CosineSimilarity(dim=0)
            sim = cos(b['features'][0], b['features'][1]).item()
            logging.info(f"Sanity Check -> Kosinová podobnost mezi kapsou 1 a 2 u ukázkového proteinu: {sim:.4f}")
            if sim > 0.999:
                logging.warning("⚠️ POZOR: Embeddingy kapes jsou téměř identické. Attention se bude učit velmi obtížně!")
            break

    in_features = train_bags[0]['features'].shape[1]
    
    all_labels = set()
    for b in bags:
        all_labels.add(b['label'].item())
    num_classes = max(all_labels) + 1
    
    model = AttentionMIL(in_features=in_features, hidden_dim=hidden_dim, 
                         num_classes=num_classes, dropout=dropout, num_heads=num_heads,
                         attention_temp=attention_temp, gated_attention=gated_attention,
                         use_self_attention=use_self_attention, self_attn_heads=self_attn_heads)
    model = model.to(device)
    
    if balance_classes:
        train_labels = [b['label'].item() for b in train_bags]
        class_counts = np.bincount(train_labels, minlength=num_classes)
        total_samples = len(train_bags)
        
        weights = []
        for count in class_counts:
            if count > 0:
                weights.append(total_samples / (num_classes * count))
            else:
                weights.append(1.0)
        
        class_weights = torch.FloatTensor(weights).to(device)
        logging.info(f"Class counts in Train: {dict(enumerate(class_counts))}")
        logging.info(f"Calculated class weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Separace parametrů: Agresivnější LR dostane pouze MIL pooling vrstva, nikoliv Transformer!
    attention_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "attention" in name and "self_attention" not in name:
            attention_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.Adam([
        {'params': other_params, 'weight_decay': weight_decay, 'lr': lr},
        {'params': attention_params, 'weight_decay': 0.0, 'lr': lr * 10}  # 10x vyšší LR pro prolomení symetrie u MIL!
    ])
    
    early_stopper = EarlyStopping(patience=patience)
    
    logging.info("\n--- Trénink MIL sítě ---")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
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
            
            loss_scaled = loss / batch_size
            loss_scaled.backward()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            if batch_count == batch_size or i == len(train_bags) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0
            
        avg_train_loss = total_train_loss / len(train_bags)
        
        # --- Validace ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for bag in val_bags:
                features = bag['features'].to(device)
                label = bag['label'].to(device)
                
                logits, A = model(features)
                val_loss = criterion(logits, label)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_bags) if len(val_bags) > 0 else avg_train_loss
        
        improved = early_stopper(avg_val_loss)
        if improved:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            
        if (epoch + 1) % 5 == 0 or epoch == 0 or early_stopper.early_stop:
            logging.info(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Best Val: {best_val_loss:.4f} | Patience: {early_stopper.counter}/{patience}")
            
        if early_stopper.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
    logging.info(f"\nNačítám nejlepší model pro vyhodnocení z {model_path}...")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logging.warning(f"Soubor s váhami {model_path} nebyl znalezen. Používám stávající váhy.")
        
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    
    if evaluate_test:
        eval_bags = test_bags
        eval_name = "Testovací sadě"
        if len(eval_bags) == 0:
            eval_bags = val_bags
            eval_name = "Validační sadě"
    else:
        eval_bags = val_bags
        eval_name = "Validační sadě"
        if len(eval_bags) == 0:
            eval_bags = train_bags
            eval_name = "Tréninkové sadě"
            
    logging.info(f"\n--- Výsledky na {eval_name} ---")
    if len(eval_bags) == 0:
        logging.error("Žádné bagy pro evaluaci nejsou k dispozici.")
        return
        
    with torch.no_grad():
        for bag in eval_bags:
            features = bag['features'].to(device)
            logits, A = model(features)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = np.argmax(probs)
            
            y_true.append(bag['label'].item())
            y_pred.append(pred)
            y_probs.append(probs)
            
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Macro F1: {f1:.4f}")
    
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        logging.info(f"ROC AUC:  {auc:.4f}")
    except ValueError:
        pass
        
    try:
        num_classes_found = len(y_probs[0])
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes_found)))
        pr_auc = average_precision_score(y_true_bin, y_probs, average='macro')
        logging.info(f"PR AUC:   {pr_auc:.4f}")
    except Exception as e:
        logging.warning(f"Nepodařilo se spočítat PR AUC: {e}")
        
    logging.info("\nClassification Report:")
    logging.info("\n" + classification_report(y_true, y_pred, zero_division=0))
    
    logging.info("\n--- Ukázka nalezených kapes (Attention) ---")
    show_limit = 5
    shown = 0
    with torch.no_grad():
        for bag in eval_bags:
            if shown >= show_limit: break
            features = bag['features'].to(device)
            logits, A = model(features)
            
            A_np = A.cpu().numpy()
            pockets = bag['pocket_ids']
            output_lines = [f"Protein: {bag['protein_id']} (True Class: {bag['label'].item()})"]
            
            if num_heads == 1:
                A_single = A_np.squeeze()
                if A_single.ndim == 0:
                    A_single = np.array([A_single.item()])
                best_idx = np.argmax(A_single)
                for i, p in enumerate(pockets):
                    marker = "<-- [VYBRÁNO MODELEM]" if i == best_idx else ""
                    output_lines.append(f"  {p}: váha {A_single[i]:.4f} {marker}")
            else:
                for i, p in enumerate(pockets):
                    weights_str = ", ".join([f"Hlava {h}: {A_np[i, h]:.4f}" for h in range(num_heads)])
                    output_lines.append(f"  {p}: {weights_str}")
                selections = []
                for h in range(num_heads):
                    best_pocket_idx = np.argmax(A_np[:, h])
                    selections.append(f"H{h} -> {pockets[best_pocket_idx]} ({A_np[best_pocket_idx, h]:.3f})")
                output_lines.append("  Rozhodnutí hlav: " + " | ".join(selections))
                
            logging.info("\n" + "\n".join(output_lines))
            shown += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention MIL classifier on EGNN embeddings")
    parser.add_argument('--embeddings-path', default='p2rank_egnn_embeddings.pt')
    parser.add_argument('--epochs', type=int, default=150, help='Max epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size via gradient accumulation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--num-heads', type=int, default=1, help='Number of attention heads (MHA-MIL)')
    parser.add_argument('--attention-temp', type=float, default=1.0, 
                        help='Softmax temperature for attention weights (values < 1.0 sharpen the distribution)')
    parser.add_argument('--gated-attention', action='store_true',
                        help='Use gated attention mechanism in MIL (Ilse et al. 2018)')
    
    # NOVÉ PARAMETRY PRO SELF-ATTENTION
    parser.add_argument('--use-self-attention', action='store_true',
                        help='Use Transformer Self-Attention layer before MIL pooling')
    parser.add_argument('--self-attn-heads', type=int, default=4,
                        help='Number of heads for Transformer Self-Attention')
    
    parser.add_argument('--balance-classes', action='store_true',
                        help='Calculate and apply balanced class weights in CrossEntropyLoss to handle imbalanced dataset')
    parser.add_argument('--model-path', default='mil_model_best.pt', help='Path to save/load best model')
    parser.add_argument('--evaluate-test', action='store_true', 
                        help='If set, evaluates on the Test set at the end; otherwise on the Val set')
    args = parser.parse_args()
    
    if not os.path.exists(args.embeddings_path):
        print(f"File {args.embeddings_path} not found.")
        exit(1)
        
    bags = load_data_and_group(args.embeddings_path)
    train_and_evaluate(
        bags=bags,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_heads=args.num_heads,
        attention_temp=args.attention_temp,
        model_path=args.model_path,
        evaluate_test=args.evaluate_test,
        gated_attention=args.gated_attention,
        balance_classes=args.balance_classes,
        use_self_attention=args.use_self_attention,
        self_attn_heads=args.self_attn_heads
    )