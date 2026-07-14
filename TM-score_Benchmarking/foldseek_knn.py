import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Foldseek KNN Benchmark (Non-ML)")
    parser.add_argument("--train-list", default="../train_mil.txt", help="Path to train IDs")
    parser.add_argument("--test-list", default="../test_mil.txt", help="Path to test IDs")
    parser.add_argument("--structures-dir", default="../structures", help="Path to structures dir containing class subfolders")
    parser.add_argument("--out-dir", default="foldseek_out", help="Directory to store intermediate foldseek files")
    parser.add_argument("--k", type=int, default=1, help="Number of nearest neighbors to consider (default 1)")
    parser.add_argument("--score-by", default="alntmscore", choices=["bits", "alntmscore", "evalue"], help="Metric to rank neighbors")
    parser.add_argument("--max-seqs", type=int, default=50, help="Max sequences passed from prefilter to alignment (default 50 to speed up tmalign)")
    parser.add_argument("--foldseek-bin", default="foldseek", help="Path to foldseek executable")
    return parser.parse_args()

def load_labels(structures_dir):
    protein_to_class = {}
    path_map = {} # Map from protein ID -> absolute path to PDB
    if os.path.exists(structures_dir):
        for cofactor in os.listdir(structures_dir):
            cofactor_path = os.path.join(structures_dir, cofactor)
            if os.path.isdir(cofactor_path):
                for filename in os.listdir(cofactor_path):
                    if filename.endswith(".pdb"):
                        prot_id = filename.replace(".pdb", "")
                        protein_to_class[prot_id] = cofactor
                        # Also support the base ID without _MERGED just in case
                        base_id = prot_id.replace("_MERGED", "")
                        if base_id not in protein_to_class:
                            protein_to_class[base_id] = cofactor
                        
                        abs_path = os.path.abspath(os.path.join(cofactor_path, filename))
                        path_map[prot_id] = abs_path
                        if base_id not in path_map:
                            path_map[base_id] = abs_path
    return protein_to_class, path_map

def prepare_db_dir(id_list, path_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(id_list, 'r') as f:
        pids = [line.strip() for line in f if line.strip()]
    
    valid_pids = []
    for pid in pids:
        if pid in path_map:
            src = path_map[pid]
            filename = os.path.basename(src)
            dst = os.path.join(out_dir, filename)
            if not os.path.exists(dst):
                os.symlink(src, dst)
            # We track the filename without .pdb because foldseek uses it as the ID
            valid_pids.append(filename.replace('.pdb', ''))
        else:
            print(f"Warning: Structure for {pid} not found in structures directory.")
    return valid_pids

def run_foldseek(test_dir, train_dir, out_tsv, tmp_dir, score_by, max_seqs, foldseek_bin="foldseek"):
    cmd = [
        foldseek_bin, "easy-search", 
        test_dir, train_dir, out_tsv, tmp_dir,
        "--max-seqs", str(max_seqs)
    ]
    
    if score_by == "alntmscore":
        # --alignment-type 1 enables TM-score alignment
        cmd.extend(["--alignment-type", "1", "--format-output", "query,target,evalue,bits,alntmscore"])
    else:
        # Default 3Di+AA alignment which is orders of magnitude faster (no TM-align)
        cmd.extend(["--format-output", "query,target,evalue,bits"])
        
    print(f"Running Foldseek: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()
    
    # 1. Load labels and paths
    print("Načítání labelů...")
    protein_to_class, path_map = load_labels(args.structures_dir)
    
    # 2. Prepare Train and Test symlink directories
    train_dir = os.path.join(args.out_dir, "train_pdbs")
    test_dir = os.path.join(args.out_dir, "test_pdbs")
    tmp_dir = os.path.join(args.out_dir, "tmp")
    results_tsv = os.path.join(args.out_dir, "results.tsv")
    
    print("Vytváření symlinků pro Foldseek...")
    train_pids = prepare_db_dir(args.train_list, path_map, train_dir)
    test_pids = prepare_db_dir(args.test_list, path_map, test_dir)
    print(f"Train set: {len(train_pids)} proteinů")
    print(f"Test set: {len(test_pids)} proteinů")
    
    # 3. Run Foldseek
    if not os.path.exists(results_tsv):
        run_foldseek(test_dir, train_dir, results_tsv, tmp_dir, args.score_by, args.max_seqs, args.foldseek_bin)
    else:
        print(f"Foldseek výsledky již existují v {results_tsv}, přeskakuji výpočet.")
        
    # 4. Parse Results and Evaluate
    print("Vyhodnocování výsledků...")
    try:
        if args.score_by == "alntmscore":
            names = ['query', 'target', 'evalue', 'bits', 'alntmscore']
        else:
            names = ['query', 'target', 'evalue', 'bits']
            
        df = pd.read_csv(results_tsv, sep='\t', header=None, names=names)
    except Exception as e:
        print(f"Chyba při čtení TSV souboru: {e}")
        return

    # Clean extensions if foldseek includes them (usually it just outputs the filename up to the first dot, or full if specified differently)
    # We strip .pdb just to be safe
    df['query'] = df['query'].str.replace('.pdb', '', regex=False)
    df['target'] = df['target'].str.replace('.pdb', '', regex=False)
    
    y_true = []
    y_pred = []
    
    # We want to group by query, sort by our metric, and pick top K
    ascending = True if args.score_by == "evalue" else False
    
    for query_id in test_pids:
        query_hits = df[df['query'] == query_id].sort_values(by=args.score_by, ascending=ascending)
        
        true_label = protein_to_class.get(query_id)
        if not true_label:
            continue
            
        if len(query_hits) == 0:
            print(f"Warning: No hits for {query_id}. Assigning 'UNKNOWN' prediction.")
            pred_label = "UNKNOWN"
        else:
            # KNN - get top K targets
            top_k_targets = query_hits.head(args.k)['target'].tolist()
            top_k_labels = [protein_to_class.get(t, "UNKNOWN") for t in top_k_targets]
            
            # Majority voting for K>1, or just the top 1
            if args.k == 1:
                pred_label = top_k_labels[0]
            else:
                pred_label = max(set(top_k_labels), key=top_k_labels.count)
                
        y_true.append(true_label)
        y_pred.append(pred_label)

    # 5. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    # Handle cases where UNKNOWN could be in y_pred
    labels = sorted(list(set(y_true)))
    
    # We calculate macro F1 only over valid classes present in y_true
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    print("\n" + "="*40)
    print("VÝSLEDKY FOLDSEEK KNN BENCHMARKU")
    print(f"Počet sousedů (K) = {args.k}")
    print(f"Metrika pro řazení = {args.score_by}")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

if __name__ == "__main__":
    main()
