import json
import os
import glob
import torch
from collections import defaultdict

def print_header(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def load_splits(prefix):
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        # Hledá soubory jako train.txt, train_e3.txt, train_mil.txt atd.
        possible_files = [f'{split_name}_{prefix}.txt', f'{split_name}.txt']
        for file_name in possible_files:
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    splits[split_name] = set(f.read().splitlines())
                break
    return splits

def analyze_raw_folders():
    print_header("0. Surová data ve složkách (PDB soubory)")
    
    # 1. EGNN (Binding_Sites)
    print("Složka Binding_Sites/PDB (Surová data pro EGNN kapsy):")
    if os.path.exists('Binding_Sites/PDB'):
        classes = [d for d in os.listdir('Binding_Sites/PDB') if os.path.isdir(os.path.join('Binding_Sites/PDB', d))]
        total_egnn = 0
        for cls in sorted(classes):
            path = os.path.join('Binding_Sites/PDB', cls, 'positive', '*.pdb')
            count = len(glob.glob(path))
            total_egnn += count
            print(f"  - {cls:<10}: {count:>5} souborů")
        print(f"  -> Celkem: {total_egnn} proteinů pro EGNN extrakci\n")
    else:
        print("  -> Složka Binding_Sites/PDB nenalezena.\n")
        
    # 2. MIL (structures)
    print("Složka structures (Surové celé proteiny pro MIL / P2Rank):")
    if os.path.exists('structures'):
        classes = [d for d in os.listdir('structures') if os.path.isdir(os.path.join('structures', d))]
        total_mil = 0
        for cls in sorted(classes):
            path = os.path.join('structures', cls, '*.pdb')
            count = len(glob.glob(path))
            total_mil += count
            print(f"  - {cls:<10}: {count:>5} souborů")
        print(f"  -> Celkem: {total_mil} proteinů pro MIL\n")
    else:
        print("  -> Složka structures nenalezena.\n")

def analyze_gt():
    print_header("1. Ground Truth (GT) pro trénink EGNN Enkodéru")
    if not os.path.exists('binding_sites_by_protein.json'):
        print("Soubor binding_sites_by_protein.json nenalezen.")
        return
        
    with open('binding_sites_by_protein.json', 'r') as f:
        data = json.load(f)
        
    num_proteins = len(data)
    # V tomto JSONu je pro každý protein jedna GT kapsa reprezentovaná jako slovník
    num_pockets = len(data)
    print(f"Celkový počet proteinů (PDB struktur): {num_proteins}")
    print(f"Celkový počet GT kapes (trénovacích grafů): {num_pockets}")
    
    # Detekce splitů
    splits = load_splits('e3')
    if splits:
        print("\nRozdělení do množin (na základě .txt souborů):")
        for split_name, ids in splits.items():
            # Některá data mají prefix 'clean_', vyřešíme obě varianty
            split_prots = [pid for pid in data.keys() if pid in ids or pid.replace('clean_', '') in ids]
            
            # Zjistíme i rozložení tříd
            class_counts = defaultdict(int)
            label_map = {0: 'NAD', 1: 'FAD', 2: 'ATP', 3: 'acetyl-CoA', 4: 'B12'}
            for pid in split_prots:
                lbl = data[pid].get('label', -1)
                if lbl in label_map:
                    c = label_map[lbl]
                else:
                    # Fallback pro starší verze JSONu
                    c = data[pid].get('ligand_name', 'Neznámá')
                class_counts[c] += 1
                
            class_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
            print(f"  - {split_name.capitalize():<10}: {len(split_prots):>5} proteinů/kapes | Třídy: [{class_str}]")

def analyze_p2rank():
    print_header("2. P2Rank dataset pro trénink MIL (Multiple Instance Learning)")
    if not os.path.exists('p2rank_pockets_dataset.json'):
        print("Soubor p2rank_pockets_dataset.json nenalezen.")
        return
        
    with open('p2rank_pockets_dataset.json', 'r') as f:
        data = json.load(f)
        
    proteins = set()
    class_counts = defaultdict(set)
    for item in data:
        pid = item['protein_id']
        proteins.add(pid)
        class_counts[item.get('bag_class', 'Neznámá třída')].add(pid)
        
    print(f"Celkový počet proteinů (bagů): {len(proteins)}")
    print(f"Celkový počet P2Rank predikovaných kapes: {len(data)}")
    
    print("\nZastoupení kofaktorových tříd (počty proteinů):")
    for cls, pids in class_counts.items():
        print(f"  - {cls:<10}: {len(pids)} proteinů")
        
    # Detekce splitů pro MIL
    splits = load_splits('mil')
    if splits:
        print("\nRozdělení do množin pro MIL trénink:")
        for split_name, ids in splits.items():
            # ID v p2rank můžou mít sufix '_MERGED'
            split_pockets = [item for item in data if item['protein_id'] in ids or item['protein_id'].replace('_MERGED', '') in ids]
            
            # Unikátní proteiny v tomto splitu
            split_prots = set(item['protein_id'] for item in split_pockets)
            
            # Počítání tříd pro tento split
            class_counts = defaultdict(int)
            # Spočítáme třídy přes unikátní proteiny (projdeme data a najdeme je)
            seen = set()
            for item in split_pockets:
                if item['protein_id'] not in seen:
                    c = item.get('bag_class', 'Neznámá')
                    class_counts[c] += 1
                    seen.add(item['protein_id'])
                    
            class_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
            
            print(f"  - {split_name.capitalize():<10}: {len(split_prots):>5} proteinů, {len(split_pockets):>6} kapes | Třídy: [{class_str}]")

def analyze_embeddings():
    print_header("3. Generované EGNN Embeddingy (vstup do MIL sítě)")
    if not os.path.exists('p2rank_egnn_embeddings.pt'):
        print("Soubor p2rank_egnn_embeddings.pt nenalezen.")
        return
        
    try:
        data = torch.load('p2rank_egnn_embeddings.pt', map_location='cpu', weights_only=False)
        embs = data['embeddings']
        protein_ids = data['protein_ids']
        unique_pids = set(protein_ids)
        
        num_embs = embs.shape[0] if hasattr(embs, 'shape') else len(embs)
        dim_embs = embs.shape[1] if hasattr(embs, 'shape') else len(embs[0])
        
        print(f"Celkový počet vygenerovaných embeddingů (kapes): {num_embs}")
        print(f"Celkový počet unikátních proteinů v embeddingech: {len(unique_pids)}")
        print(f"Dimenze jednoho embeddingu kapsy: {dim_embs}")
        
        # Načtení mapování tříd z p2rank_pockets_dataset.json (výsledného jsonu)
        pid_to_class = {}
        label_to_class = {}
        if os.path.exists('p2rank_pockets_dataset.json'):
            try:
                with open('p2rank_pockets_dataset.json', 'r', encoding='utf-8') as f:
                    p2rank_data = json.load(f)
                for item in p2rank_data:
                    pid = item.get('protein_id')
                    cls_name = item.get('bag_class') or item.get('ligand_name')
                    lbl = item.get('label')
                    if pid and cls_name:
                        pid_to_class[pid] = cls_name
                        pid_to_class[pid.replace('_MERGED', '')] = cls_name
                    if lbl is not None and cls_name:
                        label_to_class[int(lbl)] = cls_name
            except Exception as je:
                print(f"Varování: Nepodařilo se načíst mapování tříd z p2rank_pockets_dataset.json: {je}")

        # Seskupení embeddingů/kapes do bagů u reálných dat do MIL
        bags = defaultdict(list)
        bag_labels = {}
        for i, pid in enumerate(protein_ids):
            bags[pid].append(embs[i])
            bag_labels[pid] = data['labels'][i]

        # Celkové zastoupení tříd u reálných proteinů pro MIL
        class_counts = defaultdict(int)
        for pid, lbl in bag_labels.items():
            cls_name = pid_to_class.get(pid) or pid_to_class.get(pid.replace('_MERGED', '')) or label_to_class.get(int(lbl)) or f"Class {lbl}"
            class_counts[cls_name] += 1
            
        print("\nReálné zastoupení kofaktorových tříd v embeddingách:")
        for cls, count in sorted(class_counts.items()):
            print(f"  - {cls:<10}: {count} proteinů")

        # Rozdělení do splitů tak, jak to dělá train_mil_classifier.py
        # Zkusíme načíst train_mil.txt atd.
        train_ids, val_ids, test_ids = set(), set(), set()
        has_splits = False
        if os.path.exists('train_mil.txt'):
            train_ids = set(open('train_mil.txt').read().splitlines())
            has_splits = True
        if os.path.exists('validation_mil.txt'):
            val_ids = set(open('validation_mil.txt').read().splitlines())
            has_splits = True
        if os.path.exists('test_mil.txt'):
            test_ids = set(open('test_mil.txt').read().splitlines())
            has_splits = True

        if has_splits:
            print("\nRozdělení do množin pro MIL na základě *_mil.txt souborů:")
            train_bags, val_bags, test_bags = [], [], []
            for pid in bags.keys():
                pid_clean = pid.replace('_MERGED', '')
                if pid in train_ids or pid_clean in train_ids:
                    train_bags.append(pid)
                elif pid in val_ids or pid_clean in val_ids:
                    val_bags.append(pid)
                elif pid in test_ids or pid_clean in test_ids:
                    test_bags.append(pid)
                else:
                    train_bags.append(pid) # Fallback do train v train_mil_classifier.py
            
            for sname, spids in [('Train', train_bags), ('Validation', val_bags), ('Test', test_bags)]:
                scounts = defaultdict(int)
                num_pockets = 0
                for pid in spids:
                    lbl = bag_labels[pid]
                    cls_name = pid_to_class.get(pid) or pid_to_class.get(pid.replace('_MERGED', '')) or label_to_class.get(int(lbl)) or f"Class {lbl}"
                    scounts[cls_name] += 1
                    num_pockets += len(bags[pid])
                class_str = ", ".join([f"{k}: {v}" for k, v in sorted(scounts.items())])
                print(f"  - {sname:<10}: {len(spids):>5} proteinů, {num_pockets:>6} kapes | Třídy: [{class_str}]")
        else:
            # Replikace random splitu z train_mil_classifier.py (seed 42, 80% train, 20% test, 10% z train pro val)
            print("\nSplit soubory *_mil.txt nenalezeny. Replikace náhodného rozdělení (random 80/20 split jako v train_mil_classifier.py):")
            
            import numpy as np
            np_state = np.random.get_state()
            np.random.seed(42)
            
            bag_pids = list(bags.keys())
            np.random.shuffle(bag_pids)
            split_idx = int(len(bag_pids) * 0.8)
            train_bags = bag_pids[:split_idx]
            test_bags = bag_pids[split_idx:]
            
            np.random.shuffle(train_bags)
            val_idx = int(len(train_bags) * 0.1)
            val_bags = train_bags[:val_idx]
            train_bags = train_bags[val_idx:]
            
            np.random.set_state(np_state)
            
            for sname, spids in [('Train', train_bags), ('Validation', val_bags), ('Test', test_bags)]:
                scounts = defaultdict(int)
                num_pockets = 0
                for pid in spids:
                    lbl = bag_labels[pid]
                    cls_name = pid_to_class.get(pid) or pid_to_class.get(pid.replace('_MERGED', '')) or label_to_class.get(int(lbl)) or f"Class {lbl}"
                    scounts[cls_name] += 1
                    num_pockets += len(bags[pid])
                class_str = ", ".join([f"{k}: {v}" for k, v in sorted(scounts.items())])
                print(f"  - {sname:<10}: {len(spids):>5} proteinů, {num_pockets:>6} kapes | Třídy: [{class_str}]")
                
    except Exception as e:
        print(f"Nelze načíst embeddingy: {e}")

if __name__ == '__main__':
    analyze_raw_folders()
    analyze_gt()
    analyze_p2rank()
    analyze_embeddings()
    print_header("Konec reportu")
