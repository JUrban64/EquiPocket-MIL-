import json
import os
import glob
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
import tqdm

def is_aa(residue):
    return residue.get_id()[0] == ' '

def get_ca_coords_and_seq(structure):
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    ca_coords = []
    sequence = []
    residue_ids = set() # Pro porovnání overlapu (chain, resseq)
    
    # Procházíme atomy v kapse
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    resname = residue.get_resname()
                    if resname in three_to_one:
                        sequence.append(three_to_one[resname])
                        
                        chain_id = chain.get_id()
                        resseq = residue.get_id()[1]
                        residue_ids.add((chain_id, resseq))
                        
                        if 'CA' in residue:
                            ca_coords.append(residue['CA'].get_coord())
                        else:
                            coords = [a.get_coord() for a in residue.get_atoms()]
                            ca_coords.append(np.mean(coords, axis=0))
                            
    return ''.join(sequence), np.array(ca_coords).tolist(), residue_ids

def compute_contact_map(ca_coords, threshold=8.0):
    if len(ca_coords) == 0:
        return []
    coords = np.array(ca_coords)
    dist_matrix = cdist(coords, coords)
    contact_map = (dist_matrix < threshold).astype(float)
    return contact_map.tolist()

def process_p2rank_pockets(pockets_dir, output_json):
    """
    Načte PDB soubory kapes z p2ranku a převede je do JSON formátu pro EGNN/MIL grafy.
    Získá globální (bag-level) label z názvu nadřazené složky (např. NAD, FAD).
    """
    try:
        from Binding_site_ex import COFACTOR_FUNCTIONAL_GROUPS
        supported_cofactors = list(COFACTOR_FUNCTIONAL_GROUPS.keys())
    except ImportError:
        # Fallback if Binding_site_ex isn't accessible
        supported_cofactors = ['NAD', 'FAD', 'ATP', 'ADP', 'COA']
        
    pocket_files = glob.glob(os.path.join(pockets_dir, '**', '*_pocket_*.pdb'), recursive=True)
    print(f"Found {len(pocket_files)} p2rank pocket PDB files.")
    
    parser = PDBParser(QUIET=True)
    p2rank_dataset = []
    
    for pfile in tqdm.tqdm(pocket_files):
        # Extrakce protein ID a složky
        basename = os.path.basename(pfile)
        prot_id = basename.split('_pocket_')[0]
        
        # Oříznutí dalších přípon (např _prank_output, pokud tam zůstaly)
        if prot_id.endswith('_prank_output'):
            prot_id = prot_id.replace('_prank_output', '')
            
        # Zjištění labelu z cesty
        path_parts = Path(pfile).parts
        cofactor_name = None
        for cof in supported_cofactors:
            if cof in path_parts:
                cofactor_name = cof
                break
                
        if cofactor_name is None:
            # Přeskočíme, pokud nevíme, jaká je to třída
            continue
            
        label_idx = supported_cofactors.index(cofactor_name)
        
        try:
            structure = parser.get_structure('pocket', pfile)
            seq, ca_coords, pocket_res_ids = get_ca_coords_and_seq(structure)
            
            if len(seq) == 0:
                continue
                
            contact_map = compute_contact_map(ca_coords)
            
            # Ukládáme jako dict do pole
            item = {
                'pocket_id': basename,
                'protein_id': prot_id,
                'label': label_idx,
                'ligand_name': cofactor_name,
                'binding_site_sequence': seq,
                'full_sequence': seq,  # P2Rank kapsa je brána jako samostatná sekvence pro lokální ESM
                'contact_map': contact_map,
                'protein_coords': ca_coords,
                'n_binding_site': len(seq),
                'binding_site_indices': list(range(len(seq))),
                'pdb_file': pfile
            }
            
            p2rank_dataset.append(item)
            
        except Exception as e:
            print(f"Error processing {pfile}: {e}")
            
    print(f"Successfully processed {len(p2rank_dataset)} pockets.")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(p2rank_dataset, f, indent=2)
        
    print(f"Saved dataset to {output_json}")


if __name__ == "__main__":
    pockets_dir = "./structures"
    output_json = "p2rank_pockets_dataset.json"
    
    if not os.path.exists(pockets_dir):
        print(f"Directory {pockets_dir} not found. Please run prank_predict.py first.")
        exit(1)
        
    process_p2rank_pockets(pockets_dir, output_json)
