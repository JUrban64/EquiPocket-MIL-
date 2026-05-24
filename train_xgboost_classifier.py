import numpy as np
import os
import argparse
import logging
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Nastavení logování shodné s train_mil_classifier.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("xgboost_training.log"),
        logging.StreamHandler()
    ]
)

try:
    import xgboost as xgb
except ImportError:
    logging.error("Knihovna 'xgboost' není nainstalována! Nainstalujte ji spuštěním: pip install xgboost")
    logging.info("Vytvářím skript, ale před spuštěním bude nutné xgboost nainstalovat.")

def load_data_and_group(embeddings_path):
    logging.info(f"Načítám embeddingy z {embeddings_path}...")
    
    # Detekujeme příponu a případně zkonvertujeme .pt do .npz v odděleném podprocesu
    # to zamezuje OpenMP segfaultům na macOS (střet PyTorch a XGBoost OpenMP v jednom procesu)
    if embeddings_path.endswith('.pt'):
        logging.info("Detekován .pt soubor. Konvertuji jej na dočasný .npz v odděleném procesu pro zamezení OpenMP konfliktů...")
        import subprocess
        import sys
        temp_npz = "temp_embeddings_for_xgboost.npz"
        cmd = [
            sys.executable, "-c",
            f"import torch, numpy as np; "
            f"d = torch.load({repr(embeddings_path)}, weights_only=False); "
            f"np.savez({repr(temp_npz)}, "
            f"embeddings=d['embeddings'], labels=d['labels'], "
            f"protein_ids=d['protein_ids'], pocket_ids=d['pocket_ids'], "
            f"pdb_ids=d.get('pdb_ids', d['protein_ids']))"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Chyba při konverzi .pt souboru: {e.stderr}")
            raise e
        
        data = np.load(temp_npz, allow_pickle=True)
        # Smažeme dočasný soubor po načtení
        try:
            os.remove(temp_npz)
        except Exception:
            pass
    else:
        data = np.load(embeddings_path, allow_pickle=True)
    
    X = data['embeddings']      # [total_pockets, in_features]
    labels = data['labels']      # [total_pockets]
    protein_ids = data['protein_ids']
    pocket_ids = data['pocket_ids']
    
    # Seskupení do proteinů (bagů)
    bags = defaultdict(list)
    bag_labels = {}
    bag_pockets = defaultdict(list)
    
    for i, pid in enumerate(protein_ids):
        bags[pid].append(X[i])
        bag_labels[pid] = labels[i]
        bag_pockets[pid].append(pocket_ids[i])
        
    # Najdeme unikátní labely a vytvoříme mapování pro kontinuální indexy (vyžadováno XGBoostem)
    unique_labels = sorted(list(set(bag_labels.values())))
    label_map = {orig: new for new, orig in enumerate(unique_labels)}
    
    bag_list = []
    for pid in bags.keys():
        orig_label = int(bag_labels[pid])
        bag_list.append({
            'protein_id': pid,
            'features': np.stack(bags[pid]),  # [num_pockets, in_features]
            'label': label_map[orig_label],    # Kontinuální index třídy (0..C-1)
            'pocket_ids': bag_pockets[pid]
        })
        
    logging.info(f"Celkem načteno proteinů (bagů): {len(bag_list)}")
    return bag_list, unique_labels

def load_split_ids():
    train_ids, val_ids, test_ids = set(), set(), set()
    if os.path.exists('train_mil.txt'):
        train_ids = set(open('train_mil.txt').read().splitlines())
    if os.path.exists('validation_mil.txt'):
        val_ids = set(open('validation_mil.txt').read().splitlines())
    if os.path.exists('test_mil.txt'):
        test_ids = set(open('test_mil.txt').read().splitlines())
    return train_ids, val_ids, test_ids

def prepare_splits(bags):
    train_ids, val_ids, test_ids = load_split_ids()
    train_bags, val_bags, test_bags = [], [], []
    
    if train_ids and test_ids:
        logging.info("Používám rozdělení ze souborů train_mil.txt, validation_mil.txt, test_mil.txt...")
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
        logging.info("Soubory splitů nebyly nalezeny. Provádím náhodné rozdělení 80/10/10...")
        np.random.seed(42)
        np.random.shuffle(bags)
        n = len(bags)
        train_bags = bags[:int(n * 0.8)]
        val_bags = bags[int(n * 0.8):int(n * 0.9)]
        test_bags = bags[int(n * 0.9):]
        
    if len(val_bags) == 0:
        np.random.seed(42)
        np.random.shuffle(train_bags)
        val_idx = int(len(train_bags) * 0.1)
        val_bags = train_bags[:val_idx]
        train_bags = train_bags[val_idx:]
        
    logging.info(f"Sada: Train={len(train_bags)}, Val={len(val_bags)}, Test={len(test_bags)}")
    return train_bags, val_bags, test_bags

# =====================================================================
# GPU (CUDA) PODPORA PRO XGBOOST
# =====================================================================
_gpu_device = None
_gpu_tree_method = None

def init_gpu_support():
    global _gpu_device, _gpu_tree_method
    if _gpu_device is not None:
        return
        
    logging.info("Zjišťuji dostupnost GPU (CUDA) pro XGBoost...")
    # Zkusíme moderní XGBoost API (device='cuda')
    try:
        clf = xgb.XGBClassifier(device='cuda', n_estimators=1)
        clf.fit(np.random.rand(2, 2), np.array([0, 1]))
        _gpu_device = 'cuda'
        logging.info("-> CUDA GPU je plně dostupné pro XGBoost!")
        return
    except Exception:
        pass
        
    # Zkusíme starší XGBoost API (tree_method='gpu_hist')
    try:
        clf = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
        clf.fit(np.random.rand(2, 2), np.array([0, 1]))
        _gpu_tree_method = 'gpu_hist'
        logging.info("-> CUDA GPU je plně dostupné pro XGBoost (přes tree_method='gpu_hist')!")
        return
    except Exception:
        pass
        
    logging.info("-> GPU není dostupné nebo chybí knihovny pro CUDA v XGBoost. Bude použito CPU.")
    _gpu_device = 'cpu'

def create_xgb_classifier(num_classes, **kwargs):
    global _gpu_device, _gpu_tree_method
    if _gpu_device is None:
        init_gpu_support()
        
    params = kwargs.copy()
    if _gpu_device == 'cuda':
        params['device'] = 'cuda'
    elif _gpu_tree_method == 'gpu_hist':
        params['tree_method'] = 'gpu_hist'
        
    return xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        **params
    )

# =====================================================================
# PŘÍSTUP B: Protein-level (Agregace embeddingů kapes do jednoho vektoru)
# =====================================================================
def get_protein_features(bags, pooling='concat'):
    """
    Agreguje embeddingy kapes pro každý protein do jednoho vektoru.
    Podporuje: 'mean', 'max', 'concat' (spojení průměru a maxima)
    """
    X_proj = []
    y = []
    for b in bags:
        pockets = b['features']  # [num_pockets, in_features]
        if pooling == 'mean':
            feat = np.mean(pockets, axis=0)
        elif pooling == 'max':
            feat = np.max(pockets, axis=0)
        elif pooling == 'concat':
            mean_feat = np.mean(pockets, axis=0)
            max_feat = np.max(pockets, axis=0)
            feat = np.concatenate([mean_feat, max_feat])
        
        X_proj.append(feat)
        y.append(b['label'])
        
    return np.array(X_proj), np.array(y)

def tune_protein_level(X_train, y_train, X_val, y_val, num_classes, sample_weight, n_iter=15):
    logging.info(f"\n--- Spouštím Optimalizaci Hyperparametrů (Protein-level, {n_iter} iterací) ---")
    
    # Parametrický prostor pro XGBoost
    param_space = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        'n_estimators': [100, 200, 300, 400, 500],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0.0, 0.1, 0.2, 0.4]
    }
    
    best_f1 = -1.0
    best_params = None
    
    np.random.seed(42)
    tested_combinations = set()
    
    for i in range(n_iter):
        # Hledáme unikátní kombinaci (max 50 pokusů)
        for _ in range(50):
            params = {k: np.random.choice(v) for k, v in param_space.items()}
            params = {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in params.items()}
            param_tuple = tuple(sorted(params.items()))
            if param_tuple not in tested_combinations:
                tested_combinations.add(param_tuple)
                break
                
        logging.info(f"Iterace {i+1}/{n_iter}: zkouším {params}")
        
        model = create_xgb_classifier(num_classes, **params)
        
        try:
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred_probs = model.predict_proba(X_val)
            y_pred = np.argmax(y_pred_probs, axis=1)
            val_f1 = f1_score(y_val, y_pred, average='macro')
            val_acc = accuracy_score(y_val, y_pred)
            
            logging.info(f" -> Val F1: {val_f1:.4f} | Val Accuracy: {val_acc:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = params
                logging.info("   *** Nové nejlepší parametry! ***")
                
        except Exception as e:
            logging.error(f" Chyba při trénování s parametry {params}: {e}")
            
    logging.info(f"\nOptimalizace dokončena!")
    logging.info(f"Nejlepší dosažené Macro F1: {best_f1:.4f}")
    logging.info(f"Vybrané parametry: {best_params}")
    
    return best_params

def train_protein_level(train_bags, val_bags, test_bags, unique_labels, evaluate_test=False, pooling='concat', tune=False, n_iter=15):
    logging.info(f"\n--- [Přístup B] Spouštím Protein-level XGBoost (Pooling: {pooling}) ---")
    
    # 1. Příprava dat
    X_train, y_train = get_protein_features(train_bags, pooling=pooling)
    X_val, y_val = get_protein_features(val_bags, pooling=pooling)
    X_test, y_test = get_protein_features(test_bags, pooling=pooling)
    
    # Detekce tříd
    num_classes = len(unique_labels)
    logging.info(f"Počet unikátních tříd: {num_classes}")
    
    # Výpočet vah tříd (Class Weights) pro vyvážení nevyváženého datasetu
    logging.info("Vypočítávám class weights...")
    class_counts = np.bincount(y_train, minlength=num_classes)
    logging.info(f"Třídy v trénovací sadě: {class_counts}")
    total_samples = len(y_train)
    sample_weight = np.array([total_samples / (num_classes * class_counts[c]) if class_counts[c] > 0 else 1.0 for c in y_train])
    
    # 2. Inicializace a trénink XGBoost
    if tune:
        best_params = tune_protein_level(X_train, y_train, X_val, y_val, num_classes, sample_weight, n_iter=n_iter)
        logging.info("Inicializuji XGBClassifier s optimalizovanými parametry...")
        model = create_xgb_classifier(num_classes, **best_params)
    else:
        logging.info("Inicializuji XGBClassifier s výchozími parametry...")
        model = create_xgb_classifier(
            num_classes,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )
    
    logging.info("Volám model.fit()...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    logging.info("model.fit() dokončen.")
    
    # 3. Vyhodnocení
    eval_name = "Testovací sadě" if evaluate_test else "Validační sadě"
    X_eval, y_eval_true = (X_test, y_test) if evaluate_test else (X_val, y_val)
    
    y_pred_probs = model.predict_proba(X_eval)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_eval_true, y_pred)
    f1 = f1_score(y_eval_true, y_pred, average='macro')
    
    logging.info(f"\nVýsledky na {eval_name}:")
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Macro F1: {f1:.4f}")
    
    try:
        auc = roc_auc_score(y_eval_true, y_pred_probs, multi_class='ovr')
        logging.info(f"ROC AUC:  {auc:.4f}")
    except ValueError:
        pass
        
    logging.info("\nClassification Report:")
    target_names = [f"Třída {l}" for l in unique_labels]
    logging.info("\n" + classification_report(y_eval_true, y_pred, labels=list(range(num_classes)), target_names=target_names, zero_division=0))
    
    return acc, f1

# =====================================================================
# PŘÍSTUP A: Pocket-level (Predikce pro každou kapsu + následná agregace)
# =====================================================================
def get_pocket_features(bags):
    """
    Rozbalí proteiny na jednotlivé kapsy (každá kapsa dostane label celého proteinu).
    Drží si mapování, ke kterému proteinu kapsa patří.
    """
    X_pockets = []
    y_pockets = []
    pocket_to_protein = []  # Sledování, ke kterému proteinu (bagu) kapsa patří
    
    for b_idx, b in enumerate(bags):
        for pocket_feat in b['features']:
            X_pockets.append(pocket_feat)
            y_pockets.append(b['label'])
            pocket_to_protein.append(b_idx)
            
    return np.array(X_pockets), np.array(y_pockets), np.array(pocket_to_protein)

def train_pocket_level(train_bags, val_bags, test_bags, unique_labels, evaluate_test=False, agg_method='sum'):
    logging.info(f"\n--- [Přístup A] Spouštím Pocket-level XGBoost (Agregace: {agg_method}) ---")
    
    # 1. Příprava dat (rozbalení na jednotlivé kapsy)
    X_train_pockets, y_train_pockets, _ = get_pocket_features(train_bags)
    X_val_pockets, y_val_pockets, _ = get_pocket_features(val_bags)
    
    num_classes = len(unique_labels)
    
    # Výpočet vah tříd pro vyvážení
    class_counts = np.bincount(y_train_pockets, minlength=num_classes)
    total_samples = len(y_train_pockets)
    sample_weight = np.array([total_samples / (num_classes * class_counts[c]) if class_counts[c] > 0 else 1.0 for c in y_train_pockets])
    
    # 2. Inicializace a trénink XGBoost na úrovni kapes
    model = create_xgb_classifier(
        num_classes,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    model.fit(
        X_train_pockets, y_train_pockets,
        sample_weight=sample_weight,
        eval_set=[(X_val_pockets, y_val_pockets)],
        verbose=False
    )
    
    # 3. Vyhodnocení (predikce pro každou kapsu zvlášť a následná agregace pro protein)
    eval_bags = test_bags if evaluate_test else val_bags
    eval_name = "Testovací sadě" if evaluate_test else "Validační sadě"
    
    y_true_protein = []
    y_pred_protein = []
    y_probs_protein = []
    
    for b in eval_bags:
        pockets = b['features']  # [num_pockets, in_features]
        # Predikujeme pravděpodobnosti pro všechny kapsy v tomto proteinu
        pocket_probs = model.predict_proba(pockets)  # [num_pockets, num_classes]
        
        # Agregace pravděpodobností do jednoho protein-level vektoru
        if agg_method == 'sum':
            # Sečteme pravděpodobnosti všech kapes
            bag_probs = np.sum(pocket_probs, axis=0)
            # Normalizace na pravděpodobnostní rozdělení
            bag_probs = bag_probs / np.sum(bag_probs)
        elif agg_method == 'max':
            # Vezmeme maximální pravděpodobnost pro každou třídu napříč kapsami
            bag_probs = np.max(pocket_probs, axis=0)
            bag_probs = bag_probs / np.sum(bag_probs)
        elif agg_method == 'mean':
            # Průměr pravděpodobností
            bag_probs = np.mean(pocket_probs, axis=0)
            
        pred_class = np.argmax(bag_probs)
        
        y_true_protein.append(b['label'])
        y_pred_protein.append(pred_class)
        y_probs_protein.append(bag_probs)
        
    y_true_protein = np.array(y_true_protein)
    y_pred_protein = np.array(y_pred_protein)
    y_probs_protein = np.array(y_probs_protein)
    
    acc = accuracy_score(y_true_protein, y_pred_protein)
    f1 = f1_score(y_true_protein, y_pred_protein, average='macro')
    
    logging.info(f"\nVýsledky na {eval_name} (po agregaci kapes):")
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Macro F1: {f1:.4f}")
    
    try:
        auc = roc_auc_score(y_true_protein, y_probs_protein, multi_class='ovr')
        logging.info(f"ROC AUC:  {auc:.4f}")
    except ValueError:
        pass
        
    logging.info("\nClassification Report:")
    target_names = [f"Třída {l}" for l in unique_labels]
    logging.info("\n" + classification_report(y_true_protein, y_pred_protein, labels=list(range(num_classes)), target_names=target_names, zero_division=0))
    
    return acc, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost baseline on pocket embeddings")
    parser.add_argument('--embeddings-path', default='p2rank_egnn_embeddings.pt')
    parser.add_argument('--evaluate-test', action='store_true', 
                        help='Pokud je nastaveno, vyhodnocuje na Test setu; jinak na Val setu')
    parser.add_argument('--mode', default='protein', choices=['protein', 'pocket', 'both'],
                        help='Který přístup XGBoost spustit')
    parser.add_argument('--pooling', default='concat', choices=['mean', 'max', 'concat'],
                        help='Typ pooling agregace pro protein-level')
    parser.add_argument('--agg', default='sum', choices=['sum', 'max', 'mean'],
                        help='Typ agregace pravděpodobností kapes pro pocket-level')
    parser.add_argument('--tune', action='store_true',
                        help='Pokud je nastaveno, spustí optimalizaci hyperparametrů pro Metodu B')
    parser.add_argument('--n-iter', type=int, default=15,
                        help='Počet iterací pro optimalizaci hyperparametrů')
    args = parser.parse_args()
    
    if not os.path.exists(args.embeddings_path):
        logging.error(f"Soubor {args.embeddings_path} nebyl nalezen.")
        exit(1)
        
    bags, unique_labels = load_data_and_group(args.embeddings_path)
    train_bags, val_bags, test_bags = prepare_splits(bags)
    
    results = {}
    
    if args.mode in ['protein', 'both']:
        prot_acc, prot_f1 = train_protein_level(
            train_bags, val_bags, test_bags, unique_labels,
            evaluate_test=args.evaluate_test, 
            pooling=args.pooling,
            tune=args.tune,
            n_iter=args.n_iter
        )
        results['Protein-level (Approach B)'] = (prot_acc, prot_f1)
        
    if args.mode in ['pocket', 'both']:
        pock_acc, pock_f1 = train_pocket_level(
            train_bags, val_bags, test_bags, unique_labels,
            evaluate_test=args.evaluate_test, 
            agg_method=args.agg
        )
        results['Pocket-level (Approach A)'] = (pock_acc, pock_f1)
        
    # Finální hezké shrnutí
    logging.info("\n" + "="*50)
    logging.info("CELKOVÉ SHRNUTÍ XGBOOST VÝSLEDKŮ")
    logging.info("="*50)
    for model_name, (acc, f1) in results.items():
        logging.info(f"{model_name:30s} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    logging.info("="*50)
