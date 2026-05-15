import os
import json
import random
import shutil
import subprocess
import tempfile
import glob
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))

def create_alias_pdb(src_pdb, dst_pdb):
    """Vytvoří fyzickou kopii (symlinky mohou dělat problémy na HPC/v kontejnerech)."""
    shutil.copy2(src_pdb, dst_pdb)

def cluster_structures(target='both', test_limit=None):
    if target == 'binding_sites':
        pdb_roots = [os.path.join(script_dir, 'Binding_Sites')]
    elif target == 'structures':
        pdb_roots = [os.path.join(script_dir, 'structures')]
    else:
        pdb_roots = [
            os.path.join(script_dir, 'Binding_Sites'),
            os.path.join(script_dir, 'structures')
        ]
    
    pdb_files = []
    for root in pdb_roots:
        if os.path.exists(root):
            for p in glob.glob(os.path.join(root, '**', '*.pdb'), recursive=True):
                # Chceme clusterovat pouze full struktury, nikoliv výstřižky kapes
                if '_pocket' not in p and 'prank_output' not in p:
                    pdb_files.append(p)

    # Aplikace testovacího limitu
    if test_limit:
        pdb_files = pdb_files[:test_limit]
        print(f"--- Testovací režim zapnut: zpracovávám pouze {len(pdb_files)} struktur ---")

    with tempfile.TemporaryDirectory(prefix="fs_pdb_") as tmp_dir:
        pdb_data = {}
        tmp_pdb_dir = os.path.join(tmp_dir, "pdbs")
        os.makedirs(tmp_pdb_dir, exist_ok=True)
        
        for pdb_file in pdb_files:
            base_id = os.path.basename(pdb_file).replace(".pdb", "")
            
            if base_id in pdb_data:
                print(f"Warning: duplicate ID '{base_id}', skipping {pdb_file}")
                continue
                
            alias_pdb = os.path.join(tmp_pdb_dir, f"{base_id}.pdb")
            create_alias_pdb(pdb_file, alias_pdb)
            pdb_data[base_id] = alias_pdb

        print(f"Loaded {len(pdb_data)} unique PDB structures")
        if len(pdb_data) == 0:
            print("No PDBs found.")
            return None, None, None
        
        json_path = os.path.join(script_dir, '..', 'SQBCP', 'Structure_based_Cofactor_predictor', 'binding_sites_by_protein.json')
        try:
            with open(json_path, 'r') as f:
                bs_data = json.load(f)
            labels_by_pid = {}
            for k, v in bs_data.items():
                labels_by_pid[k] = str(v.get('label', '-1'))
        except Exception:
            print("Warning: Could not load binding_sites_by_protein.json, using default label '-1'")
            labels_by_pid = {}
            
        fs_out_prefix = os.path.join(tmp_dir, "fs_out")
        fs_tmp_dir = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(fs_tmp_dir, exist_ok=True)
        
        command = [
            "foldseek", "easy-cluster", 
            tmp_pdb_dir, fs_out_prefix, fs_tmp_dir,
            "--tmscore-threshold", "0.5",
            "--alignment-type", "1",
            "-c", "0.8",
            "--threads", "8"
        ]
        
        print("Running Foldseek...")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Foldseek: {e}")
            return None, None, None
        except FileNotFoundError:
            print("Error: Foldseek executable not found. Please ensure it is installed and in your PATH.")
            return None, None, None
            
        cluster_tsv = f"{fs_out_prefix}_cluster.tsv"
        clusters = {}
        if not os.path.exists(cluster_tsv):
            print(f"Error: Foldseek output {cluster_tsv} not found.")
            return None, None, None
            
        with open(cluster_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rep = parts[0].replace('.pdb', '')
                    member = parts[1].replace('.pdb', '')
                    if rep not in clusters:
                        clusters[rep] = []
                    clusters[rep].append(member)
                    
        print(f"Foldseek identified {len(clusters)} clusters.")

        total_labels = {}
        for members in clusters.values():
            for m in members:
                l = labels_by_pid.get(m, '-1')
                total_labels[l] = total_labels.get(l, 0) + 1
                
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        splits = {"train": [], "validation": [], "test": []}
        split_names = ["train", "validation", "test"]
        split_ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
        
        split_counts = {
            "train": {l: 0 for l in total_labels},
            "validation": {l: 0 for l in total_labels},
            "test": {l: 0 for l in total_labels}
        }
        
        for rep, members in sorted_clusters:
            cluster_labels = {}
            for m in members:
                l = labels_by_pid.get(m, '-1')
                cluster_labels[l] = cluster_labels.get(l, 0) + 1
                
            best_split = None
            best_score = float('inf')
            
            for split in split_names:
                score = 0
                for l in total_labels:
                    target = total_labels[l] * split_ratios[split]
                    if target == 0:
                        continue
                    current = split_counts[split][l]
                    added = cluster_labels.get(l, 0)
                    fullness = (current + added) / target
                    score += fullness ** 2 
                    
                if score < best_score:
                    best_score = score
                    best_split = split
                    
            splits[best_split].extend(members)
            for l, count in cluster_labels.items():
                split_counts[best_split][l] += count
                
        print(f"Train:      {len(splits['train'])} proteins")
        print(f"Validation: {len(splits['validation'])} proteins")
        print(f"Test:       {len(splits['test'])} proteins")
        
        return splits["train"], splits["validation"], splits["test"]

if __name__ == "__main__":
    # Nastavení argparse
    parser = argparse.ArgumentParser(description="Cluster PDB structures using Foldseek.")
    parser.add_argument("--test", action="store_true", help="Omezí počet PDB souborů na 30 pro rychlé testování.")
    parser.add_argument("--target", choices=["binding_sites", "structures", "both"], default="both", 
                        help="Co se má clustrovat: 'binding_sites' pro trénink E3, 'structures' pro MIL klasifikátor, nebo 'both'.")
    args = parser.parse_args()

    # Určení limitu na základě argumentu
    limit = 30 if args.test else None

    # Předání limitu a cíle do funkce
    train, validation, test = cluster_structures(target=args.target, test_limit=limit)

    if train is not None:
        with open(os.path.join(script_dir, 'train.txt'), 'w') as f:
            for item in train:
                f.write(f"{item}\n")    

        with open(os.path.join(script_dir, 'validation.txt'), 'w') as f:
            for item in validation:
                f.write(f"{item}\n")
        
        with open(os.path.join(script_dir, 'test.txt'), 'w') as f:
            for item in test:
                f.write(f"{item}\n")