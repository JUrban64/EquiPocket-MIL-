import os
import csv
from Bio.PDB import PDBParser

def get_sequence_from_pdb(pdb_path):
    """
    Extracts the sequence of the main chain (first chain with > 10 residues)
    from a PDB file.
    """
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return None
        
    if not structure:
        return None
        
    model = structure[0]
    protein_chain = None
    for chain in model:
        # Přeskočíme malé řetězce (např. samotné ligandy, vodu nebo krátké úseky)
        if len(list(chain.get_residues())) > 10:
            protein_chain = chain
            break
            
    if protein_chain is None:
        # Fallback na úplně první řetězec, pokud by byly všechny krátké
        for chain in model:
            protein_chain = chain
            break
            
    if protein_chain is None:
        return None
        
    sequence = []
    for residue in protein_chain.get_residues():
        # Kontrola, zda se jedná o standardní aminokyselinu (ne HETATM)
        if residue.get_id()[0] == ' ':
            resname = residue.get_resname()
            if resname in three_to_one:
                sequence.append(three_to_one[resname])
                
    return ''.join(sequence)

def main():
    structures_dir = "structures"
    output_csv = "enzyme_sequences.csv"
    
    # Cofactory a jejich odpovídající složky
    supported_cofactors = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    
    csv_data = []
    
    if not os.path.exists(structures_dir):
        print(f"Složka {structures_dir} neexistuje.")
        return
        
    print("Skenuji složku structures...")
    
    for cofactor in supported_cofactors:
        cofactor_path = os.path.join(structures_dir, cofactor)
        if not os.path.isdir(cofactor_path):
            continue
            
        print(f"Zpracovávám třídu: {cofactor}")
        for filename in os.listdir(cofactor_path):
            if filename.endswith(".pdb"):
                # Přeskočíme případné soubory kapes z P2Ranku (ty mají v názvu _pocket_)
                if "_pocket_" in filename:
                    continue
                    
                pdb_path = os.path.join(cofactor_path, filename)
                
                # ID enzymu (např. O00159_MERGED -> O00159_MERGED)
                protein_id = filename.replace(".pdb", "")
                
                # Získáme sekvenci
                sequence = get_sequence_from_pdb(pdb_path)
                
                if sequence:
                    csv_data.append({
                        "id": protein_id,
                        "sequence": sequence,
                        "class": cofactor
                    })
                else:
                    print(f"  ⚠ Varování: Nepodařilo se získat sekvenci pro {filename}")
                    
    # Uložení do CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sequence", "class"])
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"\n✓ Hotovo! Celkem úspěšně zpracováno {len(csv_data)} enzymů.")
    print(f"Výsledný soubor byl uložen jako: {output_csv}")

if __name__ == "__main__":
    main()
