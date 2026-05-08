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
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    resname = residue.get_resname()
                    if resname in three_to_one:
                        sequence.append(three_to_one[resname])
                        if 'CA' in residue:
                            ca_coords.append(residue['CA'].get_coord())
                        else:
                            coords = [a.get_coord() for a in residue.get_atoms()]
                            ca_coords.append(np.mean(coords, axis=0))
                            
    return ''.join(sequence), np.array(ca_coords).tolist()

def compute_contact_map(ca_coords, threshold=8.0):
    if len(ca_coords) == 0:
        return []
    coords = np.array(ca_coords)
    dist_matrix = cdist(coords, coords)
    contact_map = (dist_matrix < threshold).astype(float)
    return contact_map.tolist()

COFACTOR_TO_LABEL = {
    'NAD': 0,
    'FAD': 1,
    'ATP': 2,
    'ADP': 3,
    'COA': 4,
    'acetyl-CoA': 4, # Mapped to COA
    'B12': 5
}

def process_p2rank_pockets(pockets_dir, output_json):
    """
    Načte PDB soubory kapes z p2ranku a převede je do JSON formátu pro EGNN grafy.
    Labeling: Získá bag label z názvu nadřazené složky (NAD, FAD, atd.).
    Tím se připraví na Multiple Instance Learning.
    """
    pocket_files = glob.glob(os.path.join(pockets_dir, '**', '*_pocket_*.pdb'), recursive=True)
    print(f"Found {len(pocket_files)} p2rank pocket PDB files.")
    
    parser = PDBParser(QUIET=True)
    p2rank_dataset = []
    
    for pfile in tqdm.tqdm(pocket_files):
        # Path analysis
        p = Path(pfile)
        basename = p.name
        
        # Složka o 2 úrovně výš (např. structures/NAD/Q13268_MERGED_prank_output/pocket_1.pdb)
        # p.parent = Q13268_MERGED_prank_output
        # p.parent.parent = NAD
        bag_class = p.parent.parent.name
        
        # Fallback pokud náhodou nemáme strukturu složek
        if bag_class not in COFACTOR_TO_LABEL:
            bag_class = p.parent.name
            if bag_class not in COFACTOR_TO_LABEL:
                print(f"Warning: Neznámá třída pro {pfile}, přeskakuji.")
                continue
                
        label = COFACTOR_TO_LABEL[bag_class]
        
        # Extrakce protein ID
        prot_id = basename.split('_pocket_')[0]
        if prot_id.endswith('_prank_output'):
            prot_id = prot_id.replace('_prank_output', '')
            
        try:
            structure = parser.get_structure('pocket', pfile)
            seq, ca_coords = get_ca_coords_and_seq(structure)
            
            if len(seq) == 0:
                continue
                
            contact_map = compute_contact_map(ca_coords)
            
            # Ukládáme jako dict do pole
            item = {
                'pocket_id': basename,
                'protein_id': prot_id,
                'bag_class': bag_class,
                'label': label,              
                'binding_site_sequence': seq,
                'full_sequence': seq,
                'binding_site_indices': list(range(len(seq))),
                'contact_map': contact_map,
                'protein_coords': ca_coords,
                'n_binding_site': len(seq),
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
    
    process_p2rank_pockets(pockets_dir, output_json)
