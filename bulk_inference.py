import os
import sys
import csv
import glob
import torch
import numpy as np
import subprocess
import tempfile
import argparse
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

# Importy z vašich modulů
from esm2_feature_ex import ESMFeatureExtractor
from model_E3 import GraphClassifierE3
from train_mil_classifier import AttentionMIL
from Binding_site_ex import COFACTOR_FUNCTIONAL_GROUPS
from build_p2rank_dataset import get_ca_coords_and_seq, compute_contact_map
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from extract_pockets import parse_prediction_file, parse_residue_token, expand_keep_set, ResidueSelect

# Seznam kofaktorů zachovávající pořadí indexů
SUPPORTED_COFACTORS = list(COFACTOR_FUNCTIONAL_GROUPS.keys())
INT_TO_CLASS = {i: name for i, name in enumerate(SUPPORTED_COFACTORS)}

class ProteinOnlySelect(Select):
    """Pomocná třída pro Bio.PDB, která propustí pouze standardní aminokyseliny (odstraní vodu a ligandy)."""
    def accept_residue(self, residue):
        # Odstraní heterocely (H_...), vodu (W) a nechá pouze standardní aminokyseliny
        if residue.id[0].strip() == '':
            return 1
        return 0

def clean_pdb(input_path, output_path, parser):
    """Načte PDB soubor a uloží ho očištěný pouze o proteinové atomy."""
    try:
        structure = parser.get_structure('protein', input_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_path), ProteinOnlySelect())
        return True
    except Exception as e:
        print(f"   [CHYBA] Čištění PDB selhalo u {input_path}: {e}")
        return False

def run_p2rank(pdb_path, output_dir):
    """Spustí P2Rank na daném PDB souboru do specifikované složky."""
    cmd = [
        "p2rank_2.5.1/prank", "predict",
        "-c", "default",
        "-f", str(pdb_path),
        "-o", str(output_dir),
        "-t", "4"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   [CHYBA] P2Rank selhal: {e.stderr}")
        return False

def build_graph_from_pocket(pfile, esm_extractor, parser):
    """Zpracuje 1 kapsu a vrátí PyG Data objekt pro EGNN."""
    structure = parser.get_structure('pocket', pfile)
    seq, ca_coords, _ = get_ca_coords_and_seq(structure)
    
    if len(seq) == 0:
        return None
        
    contact_map = compute_contact_map(ca_coords)
    contact_map = np.asarray(contact_map, dtype=np.float32)
    
    esm_emb = esm_extractor.extract_embeddings(seq)  # [L, 1280]
    n_prot = len(seq)
    x = torch.FloatTensor(esm_emb)
    pos_tensor = torch.tensor(np.array(ca_coords, dtype=np.float32), dtype=torch.float32)
    node_type = torch.zeros(n_prot, dtype=torch.long)
    
    pp_rows, pp_cols = np.where(contact_map > 0.5)
    pp_mask = pp_rows != pp_cols
    pp_rows = pp_rows[pp_mask]
    pp_cols = pp_cols[pp_mask]
    
    EDGE_ATTR_DIM = 5
    if len(pp_rows) > 0:
        pp_weights = contact_map[pp_rows, pp_cols]
        pp_edges = np.stack([pp_rows, pp_cols], axis=1)
        pp_attr = np.zeros((len(pp_rows), EDGE_ATTR_DIM))
        pp_attr[:, 0] = pp_weights
        
        edge_index = torch.LongTensor(pp_edges).t().contiguous()
        edge_type = torch.zeros(len(pp_rows), dtype=torch.long)
        edge_attr = torch.FloatTensor(pp_attr)
    else:
        edge_list = [[i, j] for i in range(n_prot) for j in range(n_prot) if i != j]
        edge_index = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.empty((2,0), dtype=torch.long)
        edge_type = torch.zeros(len(edge_list), dtype=torch.long)
        edge_attr = torch.zeros(len(edge_list), EDGE_ATTR_DIM)
        
    return Data(
        x=x, pos=pos_tensor, edge_index=edge_index, edge_attr=edge_attr,
        edge_type=edge_type, node_type=node_type, pocket_id=os.path.basename(pfile)
    )

def main():
    parser = argparse.ArgumentParser(description="Hromadná inference s očištěním struktur a validací labelů.")
    parser.add_argument('--labels', default='protein_labels.csv', help="Cesta k CSV souboru s pravými labely")
    parser.add_argument('--structures_dir', default='.', help="Hlavní adresář obsahující struktury (ATP, B12, atd.)")
    parser.add_argument('--output', default='inference_results.csv', help="Název výsledného CSV")
    parser.add_argument('--use-self-attention', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- PMCP HROMADNÁ INFERENCE ---")
    print(f"Použité zařízení: {device}")

    # 1. Načtení ground truth labelů
    if not os.path.exists(args.labels):
        print(f"Chyba: Soubor s labely '{args.labels}' neexistuje.")
        sys.exit(1)

    protein_labels = {}
    with open(args.labels, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader) # Přeskočit hlavičku
        for row in reader:
            if row:
                protein_labels[row[0]] = int(row[1])

    print(f"Načteno {len(protein_labels)} proteinů z {args.labels}")

    # 2. Inicializace a kontrola modelů
    if not os.path.exists("gnn_model_best_e3.pt") or not os.path.exists("mil_model_best.pt"):
        print("Chyba: Chybí gnn_model_best_e3.pt nebo mil_model_best.pt v aktuální složce.")
        sys.exit(1)

    pdb_parser = PDBParser(QUIET=True)
    esm_extractor = ESMFeatureExtractor()

    # Načtení EGNN
    egnn_model = GraphClassifierE3(
        node_dim=1280, hidden_dim=64, num_attention_heads=4, num_classes=len(SUPPORTED_COFACTORS)
    ).to(device)
    state_dict = torch.load("gnn_model_best_e3.pt", map_location=device)
    egnn_model.load_state_dict(state_dict.get('model_state_dict', state_dict))
    egnn_model.eval()

    # Načtení MIL
    mil_model = AttentionMIL(in_features=64, num_classes=len(SUPPORTED_COFACTORS), use_self_attention=args.use_self_attention)
    mil_model.load_state_dict(torch.load("mil_model_best.pt", map_location='cpu'))
    mil_model.to(device)
    mil_model.eval()

    # Vyhledání všech PDB souborů v adresářové struktuře
    all_pdb_files = glob.glob(os.path.join(args.structures_dir, '**', '*.pdb'), recursive=True)
    print(f"Nalezeno celkem {len(all_pdb_files)} PDB souborů k analýze.")

    results = []
    processed_count = 0

    # 3. Hlavní inferenční smyčka
    for pdb_path in all_pdb_files:
        filename = os.path.basename(pdb_path)
        protein_id = filename.replace(".pdb", "")
        
        # Ověříme, zda pro tento protein máme True Label
        if protein_id not in protein_labels:
            continue
            
        true_label_idx = protein_labels[protein_id]
        true_label_name = INT_TO_CLASS[true_label_idx]
        
        print(f"\n[{processed_count+1}] Zpracovávám protein: {protein_id} (Očekávaný kofaktor: {true_label_name})")
        
        # Práce v temp adresáři pro izolaci čištění a P2Ranku
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_pdb_path = os.path.join(temp_dir, f"cleaned_{filename}")
            
            # Krok A: Očištění proteinu od non-protein objektů (Voda, Ligandy, Kofaktory)
            if not clean_pdb(pdb_path, cleaned_pdb_path, pdb_parser):
                continue
                
            # Krok B: Spuštění P2Rank na očištěném souboru
            p2rank_out = os.path.join(temp_dir, "p2rank_output")
            if not run_p2rank(cleaned_pdb_path, p2rank_out):
                continue
                
            # Extrakce kapes z P2Rank predikce do PDB souborů
            pred_csvs = glob.glob(os.path.join(p2rank_out, '**', '*.pdb_predictions.csv'), recursive=True)
            if not pred_csvs:
                print(f"   -> P2Rank nenašel žádné predikce (CSV nenalezeno).")
                results.append([protein_id, true_label_name, "ŽÁDNÉ_KAPSY", 0.0, "CHYBA", "N/A", 0.0])
                processed_count += 1
                continue
                
            pockets = parse_prediction_file(pred_csvs[0])
            structure_for_pockets = pdb_parser.get_structure('protein', cleaned_pdb_path)
            
            for pid, score, residues in pockets:
                if score <= 0.5:
                    continue
                keep = set()
                for tok in residues:
                    for ch, rs in parse_residue_token(tok):
                        keep.add((ch, rs))
                keep = expand_keep_set(structure_for_pockets, keep, seq_window=4, ca_cutoff=8.0)
                if not keep:
                    continue
                out_file = os.path.join(temp_dir, f"{filename}_pocket_{pid}.pdb")
                io = PDBIO()
                io.set_structure(structure_for_pockets)
                io.save(out_file, ResidueSelect(keep))
                
            pocket_files = glob.glob(os.path.join(temp_dir, f"{filename}_pocket_*.pdb"))
            if not pocket_files:
                print(f"   -> P2Rank nenašel žádné validní kapsy na očištěném proteinu.")
                results.append([protein_id, true_label_name, "ŽÁDNÉ_KAPSY", 0.0, "CHYBA", "N/A", 0.0])
                processed_count += 1
                continue
            
            # Krok C: Extrakce ESM embeddings a tvorba grafů kapes
            graphs = []
            for pfile in pocket_files:
                g = build_graph_from_pocket(pfile, esm_extractor, pdb_parser)
                if g is not None:
                    graphs.append(g.to(device))
                    
            if not graphs:
                results.append([protein_id, true_label_name, "CHYBA_GRAFU", 0.0, "CHYBA", "N/A", 0.0])
                processed_count += 1
                continue
                
            # Krok D: EGNN Inference
            pocket_embeddings = []
            with torch.no_grad():
                for g in graphs:
                    loader = DataLoader([g], batch_size=1)
                    for batch in loader:
                        emb = egnn_model.get_embedding(batch)
                        pocket_embeddings.append(emb.squeeze(0))
                        
            # Krok E: MIL Inference
            bag_features = torch.stack(pocket_embeddings).to(device)
            with torch.no_grad():
                logits, attention_weights = mil_model(bag_features)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                
                pred_idx = np.argmax(probs)
                pred_label_name = INT_TO_CLASS[pred_idx]
                confidence = probs[pred_idx]
                
                # Zjištění nejdůležitější kapsy podle Attention
                att_weights = attention_weights.squeeze().cpu().numpy()
                if att_weights.ndim == 0:
                    att_weights = np.array([att_weights.item()])
                best_pocket_idx = np.argmax(att_weights)
                best_pocket_id = graphs[best_pocket_idx].pocket_id
                best_pocket_weight = att_weights.flatten()[best_pocket_idx]

            # Vyhodnocení správnosti predikce
            is_correct = "TRUE" if pred_idx == true_label_idx else "FALSE"
            
            print(f"   -> Predikováno: {pred_label_name} ({confidence*100:.1f}%) | Výsledek: {is_correct}")
            print(f"   -> Nejdůležitější vazebné místo: {best_pocket_id} (váha: {best_pocket_weight:.4f})")
            
            results.append([
                protein_id, 
                true_label_name, 
                pred_label_name, 
                round(float(confidence), 4), 
                is_correct,
                best_pocket_id,
                round(float(best_pocket_weight), 4)
            ])
            processed_count += 1

    # 4. Zápis finálního výsledného CSV souboru
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "protein_id", "true_label", "predicted_label", 
            "confidence", "is_correct", "best_pocket_id", "pocket_attention_weight"
        ])
        writer.writerows(results)

    print("\n" + "="*50)
    print(f"Inference dokončena. Zpracováno proteinů: {processed_count}")
    print(f"Výsledky uloženy do souboru: {args.output}")
    print("="*50)

if __name__ == '__main__':
    main()