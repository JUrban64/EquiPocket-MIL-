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

def process_p2rank_pockets(pockets_dir, gt_json_path, output_json, overlap_threshold=0.3):
    """
    Načte PDB soubory kapes z p2ranku a převede je do JSON formátu pro EGNN grafy.
    Labeling: Zjistí překryv s Ground Truth residui.
    """
    print(f"Loading Ground Truth from {gt_json_path}...")
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    # Připravíme si GT lookup table pro rychlé hledání
    gt_lookup = {}
    for prot_id, info in gt_data.items():
        gt_res = set()
        for r in info.get('binding_site_residues', []):
            gt_res.add((r['chain_id'], r['resseq']))
        gt_lookup[prot_id] = {
            'residues': gt_res,
            'label': info.get('label', -1),
            'full_sequence': info.get('full_sequence', '')
        }
    
    pocket_files = glob.glob(os.path.join(pockets_dir, '**', '*_pocket_*.pdb'), recursive=True)
    print(f"Found {len(pocket_files)} p2rank pocket PDB files.")
    
    parser = PDBParser(QUIET=True)
    p2rank_dataset = []
    
    for pfile in tqdm.tqdm(pocket_files):
        # Extrakce protein ID z názvu souboru (předpoklad: protID_pocket_X.pdb)
        basename = os.path.basename(pfile)
        prot_id = basename.split('_pocket_')[0]
        
        # Oříznutí dalších přípon (např _prank_output, pokud tam zůstaly)
        if prot_id.endswith('_prank_output'):
            prot_id = prot_id.replace('_prank_output', '')
            
        if prot_id not in gt_lookup:
            continue
            
        gt_info = gt_lookup[prot_id]
        
        try:
            structure = parser.get_structure('pocket', pfile)
            seq, ca_coords, pocket_res_ids = get_ca_coords_and_seq(structure)
            
            if len(seq) == 0:
                continue
                
            contact_map = compute_contact_map(ca_coords)
            
            # Výpočet překryvu (Intersection over Union)
            intersection = pocket_res_ids.intersection(gt_info['residues'])
            if len(pocket_res_ids) == 0:
                continue
                
            # Můžeme počítat overlap vůči velikosti kapsy, nebo vůči sjednocení
            overlap_ratio = len(intersection) / len(pocket_res_ids)
            
            # Label = 1 pokud je to správná kapsa (překryv > threshold), jinak 0
            is_correct_pocket = 1 if overlap_ratio > overlap_threshold else 0
            
            # Ukládáme jako dict do pole
            item = {
                'pocket_id': basename,
                'protein_id': prot_id,
                'is_correct_pocket': is_correct_pocket, 
                'gt_enzyme_class': gt_info['label'],    
                'label': gt_info['label'],              
                'overlap_ratio': overlap_ratio,
                'binding_site_sequence': seq,
                'full_sequence': gt_info['full_sequence'],
                'contact_map': contact_map,
                'protein_coords': ca_coords,
                'n_binding_site': len(seq),
        
            }
            
            item['full_sequence'] = seq
            item['binding_site_indices'] = list(range(len(seq)))
            item['pdb_file'] = pfile
            
            p2rank_dataset.append(item)
            
        except Exception as e:
            print(f"Error processing {pfile}: {e}")
            
    print(f"Successfully processed {len(p2rank_dataset)} pockets.")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(p2rank_dataset, f, indent=2)
        
    print(f"Saved dataset to {output_json}")


if __name__ == "__main__":
    gt_json = "binding_sites_by_protein.json"
    
    if not os.path.exists(gt_json):
        print(f"{gt_json} nenalezen. Spouštím extrakci Ground Truth přes Binding_site_ex.py...")
        import subprocess
        cmd = [
            "python", "Binding_site_ex.py",
            "--pdb-root", "Binding_Sites/PDB",
            "--output-json", gt_json
        ]
        try:
            subprocess.run(cmd, check=True)
            print("Ground Truth extrakce úspěšně dokončena.")
        except subprocess.CalledProcessError as e:
            print(f"Chyba při generování Ground Truth: {e}")
            exit(1)
            
    pockets_dir = "./structures"
    output_json = "p2rank_pockets_dataset.json"
    
    process_p2rank_pockets(pockets_dir, gt_json, output_json)
