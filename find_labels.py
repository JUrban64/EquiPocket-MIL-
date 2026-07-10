import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Použití: python find_labels.py <cesta_k_txt_souboru>")
        sys.exit(1)
        
    txt_file = sys.argv[1]
    structures_dir = "structures"
    
    if not os.path.exists(txt_file):
        print(f"Soubor {txt_file} neexistuje.")
        sys.exit(1)
        
    # Načteme ID proteinů
    with open(txt_file, 'r') as f:
        protein_ids = [line.strip() for line in f if line.strip()]
        
    # Vytvoříme mapování z protein_id -> třída (kofaktor)
    protein_to_class = {}
    
    if os.path.exists(structures_dir):
        for cofactor in os.listdir(structures_dir):
            cofactor_path = os.path.join(structures_dir, cofactor)
            if os.path.isdir(cofactor_path):
                # Projdeme všechny soubory a složky uvnitř třídy
                for filename in os.listdir(cofactor_path):
                    if filename.endswith(".pdb"):
                        prot_id = filename.replace(".pdb", "")
                        protein_to_class[prot_id] = cofactor
                        protein_to_class[prot_id.replace("_MERGED", "")] = cofactor
    
    print(f"Nalezeno {len(protein_ids)} proteinů v souboru {txt_file}.")
    print("-" * 40)
    
    found_counts = {}
    not_found = []
    
    import csv
    
    csv_output = txt_file.replace(".txt", "_labels.csv")
    
    # Mapování podle Binding_site_ex.py
    class_to_int = {
        'NAD': 0,
        'FAD': 1,
        'ATP': 2,
        'acetyl-CoA': 3,
        'B12': 4
    }
    
    csv_data = []
    for pid in protein_ids:
        if pid in protein_to_class:
            label = protein_to_class[pid]
        elif pid + "_MERGED" in protein_to_class:
            label = protein_to_class[pid + "_MERGED"]
        elif pid.replace("_MERGED", "") in protein_to_class:
            label = protein_to_class[pid.replace("_MERGED", "")]
        else:
            label = "NENALEZENO"
            
        print(f"{pid} -> {label}")
        
        if label != "NENALEZENO":
            found_counts[label] = found_counts.get(label, 0) + 1
            csv_data.append([pid, class_to_int[label]])
        else:
            not_found.append(pid)
            
    print("-" * 40)
    print("Shrnutí:")
    for label, count in sorted(found_counts.items()):
        print(f"  {label}: {count}")
    
    if not_found:
        print(f"\nUpozornění: {len(not_found)} proteinů nebylo nalezeno ve složkách.")
        
    # Uložení do CSV
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["protein_id", "true_label"])
        writer.writerows(csv_data)
        
    print(f"\nVýsledky byly uloženy do souboru: {csv_output}")

if __name__ == "__main__":
    main()
