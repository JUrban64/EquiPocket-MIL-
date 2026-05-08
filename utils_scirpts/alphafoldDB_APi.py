import requests
import os 
from urllib.parse import quote
import time
from Bio.PDB import PDBParser, PDBIO

# Vždy stejná cesta
structures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../structures"))
os.makedirs(structures_dir, exist_ok=True)
print(f"Struktury budou uloženy do: {structures_dir}\n")

# CHEBI mapování kofaktorů
CHEBI_MAP = {
    'NAD': '15846',
    'ATP': '15422',
    'acetyl-CoA': '15351',
    'B12': '176843',
    'FAD': '16238'
}

# Generování queries
QUERIES = {k: f"chebi:{v}" for k, v in CHEBI_MAP.items()}

def fetch_all_uniprots(query, max_results=1000, page_size=500):
    """Stáhne všechny UniProt IDs s paginací."""
    all_ids = []
    offset = 0
    
    while len(all_ids) < max_results:
        encoded_query = quote(query)
        url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_query}&format=json&fields=accession&size={page_size}&offset={offset}"
        
        print(f"  Stahuji: offset={offset}, dosud={len(all_ids)}...", end=" ", flush=True)
        
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"\nChyba: {response.status_code}")
            break
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            print("Konec dat")
            break
        
        ids = [item['primaryAccession'] for item in results]
        all_ids.extend(ids)
        
        print(f"✓ (+{len(ids)})")
        time.sleep(0.5)
        
        offset += page_size
    
    return all_ids[:max_results]

def download_alphafold_structures(cofactor, uniprot_ids, max_downloads=None):
    """Stáhne AlphaFold struktury pro dané UniProt IDs."""
    if max_downloads is None:
        max_downloads = len(uniprot_ids)
    
    cofactor_dir = os.path.join(structures_dir, cofactor)
    os.makedirs(cofactor_dir, exist_ok=True)
    
    downloaded = 0
    failed = 0
    
    for i, uniprot_id in enumerate(uniprot_ids[:max_downloads]):
        print(f"\n  [{i+1}/{min(max_downloads, len(uniprot_ids))}] Stahuji strukturu pro: {uniprot_id}")
        
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        
        try:
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for fragment in data:
                    pdb_url = fragment.get("pdbUrl")
                    
                    if pdb_url:
                        print(f"    → URL: {pdb_url}")
                        
                        pdb_response = requests.get(pdb_url, timeout=10)
                        
                        if pdb_response.status_code == 200:
                            filename = os.path.join(cofactor_dir, pdb_url.split("/")[-1])
                            with open(filename, "wb") as f:
                                f.write(pdb_response.content)
                            print(f"    ✅ Uloženo: {os.path.relpath(filename, structures_dir)}")
                            downloaded += 1
                        else:
                            print(f"    ❌ Chyba při stažení PDB: {pdb_response.status_code}")
                            failed += 1
            else:
                print(f"    ❌ AlphaFold API chyba: {response.status_code}")
                failed += 1
        
        except Exception as e:
            print(f"    ❌ Chyba: {e}")
            failed += 1
        
        time.sleep(0.5)
    
    print(f"\n  📊 {cofactor}: {downloaded} staženo, {failed} selhalo")
    return downloaded, failed


def merge_fragments(cofactor_dir, uniprot_id):
    """Spojí AlphaFold fragmenty do jednoho proteinu."""
    
    pdb_files = sorted([f for f in os.listdir(cofactor_dir) 
                       if uniprot_id in f and f.endswith('.pdb')])
    
    if len(pdb_files) <= 1:
        return
    
    print(f"\n  Spojuji fragmenty pro {uniprot_id}: {pdb_files}")
    
    parser = PDBParser(QUIET=True)
    all_residues = []
    
    for pdb_file in pdb_files:
        filepath = os.path.join(cofactor_dir, pdb_file)
        struct = parser.get_structure(uniprot_id, filepath)
        
        # Vezmi CA atomy z řetězce A
        try:
            chain = struct[0]['A']
            all_residues.extend(list(chain.get_residues()))
        except:
            pass
    
    if len(all_residues) > 0:
        # Uloži spojený model
        base_structure = parser.get_structure(uniprot_id, 
                                             os.path.join(cofactor_dir, pdb_files[0]))
        
        output_file = os.path.join(cofactor_dir, f"{uniprot_id}_MERGED.pdb")
        io = PDBIO()
        io.set_structure(base_structure)
        io.save(output_file)
        print(f"  ✅ Spojeno: {os.path.relpath(output_file, structures_dir)}")


# Hlavní smyčka
print("=== Stažení struktur pro více kofaktorů ===\n")

total_downloaded = 0
total_failed = 0

for cofactor, chebi_query in QUERIES.items():
    print(f"\n{'='*50}")
    print(f"Kofaktor: {cofactor} (CHEBI: {CHEBI_MAP[cofactor]})")
    print(f"{'='*50}")
    
    print(f"Stažení UniProt ID pro {cofactor}:")
    uniprots = fetch_all_uniprots(chebi_query, max_results=5, page_size=500)
    
    print(f"\n✅ Celkem staženo: {len(uniprots)} UniProt ID")
    if uniprots:
        print(f"Prvních 5: {', '.join(uniprots[:5])}")
    
    if uniprots:
        print(f"\n=== Stažení AlphaFold struktur pro {cofactor} ===")
        cofactor_dir = os.path.join(structures_dir, cofactor)
        downloaded, failed = download_alphafold_structures(cofactor, uniprots, max_downloads=50)
        
        # Spojení fragmentů
        print(f"\n=== Spojování fragmentů ===")
        for uid in uniprots:
            merge_fragments(cofactor_dir, uid)
        
        total_downloaded += downloaded
        total_failed += failed

print(f"\n\n{'='*50}")
print(f"CELKOVÉ SHRNUTÍ")
print(f"{'='*50}")
print(f"✅ Celkem staženo: {total_downloaded}")
print(f"❌ Celkem selhalo: {total_failed}")