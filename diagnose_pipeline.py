import os
import glob
import json
import torch

def main():
    structures_dir = "structures"
    dataset_json = "p2rank_pockets_dataset.json"
    embeddings_file = "p2rank_egnn_embeddings.pt"
    
    supported_cofactors = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    
    print("=== PIPELINE DIAGNOSTIC REPORT ===")
    
    # 1. Scan raw structures
    if not os.path.exists(structures_dir):
        print(f"ERROR: {structures_dir} directory not found.")
        return
        
    raw_proteins = {} # id -> (cofactor, path)
    for cofactor in supported_cofactors:
        cofactor_path = os.path.join(structures_dir, cofactor)
        if not os.path.isdir(cofactor_path):
            continue
        for filename in os.listdir(cofactor_path):
            if filename.endswith(".pdb") and "_pocket_" not in filename:
                prot_id = filename.replace(".pdb", "")
                raw_proteins[prot_id] = (cofactor, os.path.join(cofactor_path, filename))
                
    total_raw = len(raw_proteins)
    print(f"\n1. Raw proteins in structures/ folder: {total_raw}")
    for cof in supported_cofactors:
        count = sum(1 for pid, (c, _) in raw_proteins.items() if c == cof)
        print(f"   - {cof}: {count}")
        
    # 2. Check P2Rank outputs
    no_prank_output_dir = []
    zero_pockets = []
    has_pockets = {} # id -> pocket_files
    
    for prot_id, (cof, pdb_path) in raw_proteins.items():
        # Check potential prank output directories
        prank_dir1 = pdb_path.replace(".pdb", "_prank_output")
        prank_dir2 = os.path.join(os.path.dirname(pdb_path), prot_id + "_prank_output")
        
        prank_dir = prank_dir1 if os.path.isdir(prank_dir1) else (prank_dir2 if os.path.isdir(prank_dir2) else None)
        
        if not prank_dir:
            no_prank_output_dir.append((prot_id, cof))
        else:
            # Look for pocket PDB files
            pockets = glob.glob(os.path.join(prank_dir, "*_pocket_*.pdb"))
            if not pockets:
                zero_pockets.append((prot_id, cof))
            else:
                has_pockets[prot_id] = pockets
                
    print(f"\n2. P2Rank Prediction Status:")
    print(f"   - Proteins with pockets found: {len(has_pockets)}")
    print(f"   - Proteins WITHOUT P2Rank output folder: {len(no_prank_output_dir)}")
    print(f"   - Proteins with folder but 0 pockets predicted: {len(zero_pockets)}")
    
    if no_prank_output_dir:
        print("\n   [!] First 5 proteins missing P2Rank output folder:")
        for pid, cof in no_prank_output_dir[:5]:
            print(f"       - {pid} ({cof})")
            
    if zero_pockets:
        print("\n   [!] First 5 proteins with 0 predicted pockets:")
        for pid, cof in zero_pockets[:5]:
            print(f"       - {pid} ({cof})")
            
    # 3. Check JSON dataset
    json_proteins = set()
    json_pockets_count = 0
    if os.path.exists(dataset_json):
        try:
            with open(dataset_json, 'r') as f:
                data = json.load(f)
            json_pockets_count = len(data)
            for item in data:
                pid = item['protein_id']
                json_proteins.add(pid)
                json_proteins.add(pid.replace('_MERGED', ''))
            print(f"\n3. JSON Dataset ({dataset_json}):")
            print(f"   - Total unique proteins in JSON: {len(json_proteins)}")
            print(f"   - Total pockets in JSON: {json_pockets_count}")
            
            # Missing from JSON but had pockets
            missing_from_json = []
            for pid in has_pockets:
                clean_pid = pid.replace('_MERGED', '')
                if pid not in json_proteins and clean_pid not in json_proteins:
                    missing_from_json.append(pid)
            
            if missing_from_json:
                print(f"   - [!] {len(missing_from_json)} proteins had pockets but are NOT in JSON (likely Bio.PDB parsing errors or empty sequences).")
                print("     First 5 examples:")
                for pid in missing_from_json[:5]:
                    print(f"       - {pid}")
        except Exception as e:
            print(f"\n3. JSON Dataset: ERROR reading JSON: {e}")
    else:
        print(f"\n3. JSON Dataset: {dataset_json} NOT found.")
        
    # 4. Check Embeddings
    emb_proteins = set()
    if os.path.exists(embeddings_file):
        try:
            emb_data = torch.load(embeddings_file, map_location='cpu', weights_only=False)
            if 'protein_ids' in emb_data:
                for pid in emb_data['protein_ids']:
                    emb_proteins.add(pid)
                    emb_proteins.add(pid.replace('_MERGED', ''))
                print(f"\n4. Embeddings ({embeddings_file}):")
                print(f"   - Total unique proteins: {len(emb_proteins)}")
                print(f"   - Total pocket embeddings: {len(emb_data['embeddings'])}")
                
                # Missing from embeddings but were in JSON
                if json_proteins:
                    missing_from_emb = []
                    for pid in json_proteins:
                        clean_pid = pid.replace('_MERGED', '')
                        if pid not in emb_proteins and clean_pid not in emb_proteins:
                            # Avoid duplicates from clean/merged mapping
                            if clean_pid not in [x.replace('_MERGED', '') for x in missing_from_emb]:
                                missing_from_emb.append(pid)
                    if missing_from_emb:
                        print(f"   - [!] {len(missing_from_emb)} proteins are in JSON but missing in embeddings (likely skipped by splits manifest).")
                        print("     First 5 examples:")
                        for pid in missing_from_emb[:5]:
                            print(f"       - {pid}")
            else:
                print(f"\n4. Embeddings: 'protein_ids' key not found in saved embeddings.")
        except Exception as e:
            print(f"\n4. Embeddings: ERROR reading embeddings: {e}")
    else:
        print(f"\n4. Embeddings: {embeddings_file} NOT found.")
        
    print("\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Raw -> Pockets: Lost {total_raw - len(has_pockets)} proteins (either P2Rank not run or 0 pockets predicted).")
    if os.path.exists(dataset_json):
        print(f"Pockets -> JSON: Lost {len(has_pockets) - len(json_proteins)} proteins due to parser/empty-seq filtering.")
    if os.path.exists(embeddings_file):
        print(f"JSON -> Embeddings: Lost {len(json_proteins) - len(emb_proteins)} proteins (likely manifest split selection).")

if __name__ == "__main__":
    main()
