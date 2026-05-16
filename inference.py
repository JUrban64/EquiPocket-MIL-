import os
import sys
import glob
import torch
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from Bio.PDB import PDBParser
from torch_geometric.data import Data

# Import z tvých existujících skriptů
from esm2_feature_ex import ESMFeatureExtractor
from model_E3 import GraphClassifierE3
from train_mil_classifier import AttentionMIL
from Binding_site_ex import COFACTOR_FUNCTIONAL_GROUPS

# Reusing coordinate extraction logic from build_p2rank_dataset
from build_p2rank_dataset import get_ca_coords_and_seq, compute_contact_map

# Převod label zpět na text
SUPPORTED_COFACTORS = list(COFACTOR_FUNCTIONAL_GROUPS.keys())

def run_p2rank(pdb_path, output_dir):
    """Spustí P2Rank na daném PDB souboru do specifikované složky."""
    print(f"-> Spouštím P2Rank na {pdb_path}...")
    cmd = [
        "p2rank_2.5.1/prank", "predict",
        "-c", "alphafold",
        "-f", str(pdb_path),
        "-o", str(output_dir),
        "-t", "4"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   P2Rank úspěšně dokončen.")
    except subprocess.CalledProcessError as e:
        print(f"Chyba při spouštění P2Rank: {e.stderr}")
        sys.exit(1)

def build_graph_from_pocket(pfile, esm_extractor, parser):
    """Zpracuje 1 kapsu a vrátí PyG Data objekt pro EGNN."""
    structure = parser.get_structure('pocket', pfile)
    seq, ca_coords, _ = get_ca_coords_and_seq(structure)
    
    if len(seq) == 0:
        return None
        
    contact_map = compute_contact_map(ca_coords)
    contact_map = np.asarray(contact_map, dtype=np.float32)
    
    # 1. Získání ESM embeddingu pro sekvenci kapsy
    esm_emb = esm_extractor.extract_embeddings(seq)  # [L, 1280]
    
    n_prot = len(seq)
    x = torch.FloatTensor(esm_emb)
    pos_tensor = torch.tensor(np.array(ca_coords, dtype=np.float32), dtype=torch.float32)
    node_type = torch.zeros(n_prot, dtype=torch.long)
    
    # 2. Hrany z kontaktní mapy
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
        # Fallback fully connected
        edge_list = [[i, j] for i in range(n_prot) for j in range(n_prot) if i != j]
        edge_index = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.empty((2,0), dtype=torch.long)
        edge_type = torch.zeros(len(edge_list), dtype=torch.long)
        edge_attr = torch.zeros(len(edge_list), EDGE_ATTR_DIM)
        
    graph = Data(
        x=x,
        pos=pos_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        node_type=node_type,
        pocket_id=os.path.basename(pfile)
    )
    return graph

def main(pdb_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- PMCP INFERENCE PIPELINE ---")
    print(f"Zařízení: {device}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. P2Rank
        run_p2rank(pdb_path, temp_dir)
        
        pocket_files = glob.glob(os.path.join(temp_dir, '**', '*_pocket_*.pdb'), recursive=True)
        if not pocket_files:
            print("P2Rank nenašel žádné kapsy!")
            sys.exit(0)
            
        print(f"-> Nalezeno {len(pocket_files)} kapes. Extrahuji ESM a stavím grafy...")
        
        # 2. Extrakce a grafy
        esm_extractor = ESMFeatureExtractor()
        parser = PDBParser(QUIET=True)
        graphs = []
        pocket_ids = []
        
        for pfile in pocket_files:
            g = build_graph_from_pocket(pfile, esm_extractor, parser)
            if g is not None:
                g = g.to(device)
                graphs.append(g)
                pocket_ids.append(g.pocket_id)
                
        if not graphs:
            print("Nepodařilo se postavit žádné validní grafy.")
            sys.exit(0)
            
        # 3. EGNN Encoder
        print("-> Načítám EGNN model pro kódování kapes...")
        egnn_model = GraphClassifierE3(
            node_dim=1280, hidden_dim=64, num_attention_heads=4, num_classes=len(SUPPORTED_COFACTORS)
        ).to(device)
        
        if not os.path.exists("gnn_model_best_e3.pt"):
            print("Chyba: Soubor gnn_model_best_e3.pt nebyl nalezen. Natrénuj EGNN.")
            sys.exit(1)
            
        state_dict = torch.load("gnn_model_best_e3.pt", map_location=device)
        egnn_model.load_state_dict(state_dict.get('model_state_dict', state_dict))
        egnn_model.eval()
        
        print("-> Vytvářím embeddingy kapes...")
        pocket_embeddings = []
        with torch.no_grad():
            for g in graphs:
                # EGNN bere batch, proto uměle vytvoříme batch atribut pomocí .unsqueeze / DataLoader logic
                # Nejsnazší je obalit to do DataLoadery s batch_size=1
                from torch_geometric.loader import DataLoader
                loader = DataLoader([g], batch_size=1)
                for batch in loader:
                    emb = egnn_model.get_embedding(batch)
                    pocket_embeddings.append(emb.cpu().squeeze(0))
        
        # 4. MIL Klasifikátor
        print("-> Načítám MIL model pro finální klasifikaci...")
        bag_features = torch.stack(pocket_embeddings) # [num_pockets, 64]
        in_features = bag_features.shape[1]
        
        mil_model = AttentionMIL(in_features=in_features, num_classes=len(SUPPORTED_COFACTORS))
        if not os.path.exists("mil_model_best.pt"):
            print("Chyba: Soubor mil_model_best.pt nebyl nalezen. Natrénuj MIL klasifikátor.")
            sys.exit(1)
            
        mil_model.load_state_dict(torch.load("mil_model_best.pt", map_location='cpu'))
        mil_model.eval()
        
        with torch.no_grad():
            logits, attention_weights = mil_model(bag_features)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            pred_idx = np.argmax(probs)
            pred_class = SUPPORTED_COFACTORS[pred_idx]
            confidence = probs[pred_idx] * 100
            
            att_weights = attention_weights.squeeze().numpy()
            if att_weights.ndim == 0:
                att_weights = [att_weights.item()]
                
        # 5. Výpis
        print("\\n" + "="*50)
        print("                VÝSLEDEK INFERENCE")
        print("="*50)
        print(f"PDB Soubor:       {os.path.basename(pdb_path)}")
        print(f"Predikce Kofaktoru: {pred_class} (Jistota: {confidence:.1f}%)")
        print("-" * 50)
        print("Nalezené kapsy a jejich důležitost (Attention):")
        
        # Seřadit kapsy podle váhy
        sorted_indices = np.argsort(att_weights)[::-1]
        for i in sorted_indices:
            marker = "   <-- [Vazebné Místo]" if i == sorted_indices[0] else ""
            print(f" - {pocket_ids[i]}:  {att_weights[i]:.4f} {marker}")
        print("="*50 + "\\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inference pipeline for cofactor prediction")
    parser.add_argument('pdb_file', help="Cesta k PDB souboru (např. muj_protein.pdb)")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdb_file):
        print(f"Soubor {args.pdb_file} neexistuje!")
        sys.exit(1)
        
    main(args.pdb_file)
