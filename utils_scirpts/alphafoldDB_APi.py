import requests
import os 
from urllib.parse import quote
import time
import json
import datetime
from Bio.PDB import PDBParser, PDBIO

# Vždy stejná cesta
structures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../structures"))
os.makedirs(structures_dir, exist_ok=True)
log_file = os.path.join(structures_dir, "run_log.txt")

def write_log(msg, print_to_console=True, write_to_file=True):
    """Vypíše zprávu do konzole a zároveň ji uloží do logu s časovým razítkem."""
    if print_to_console:
        print(msg)
    if write_to_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            # Nechceme časové razítko u formátovacích oddělovačů (====)
            if msg.startswith("=") or msg.strip() == "":
                f.write(f"{msg}\n")
            else:
                f.write(f"[{timestamp}] {msg}\n")

write_log("="*60)
write_log("SPUŠTĚNÍ PIPELINE PRO STAŽENÍ STRUKTUR")
write_log("="*60)
write_log(f"Cílová složka: {structures_dir}")

# CHEBI mapování kofaktorů
CHEBI_MAP = {
    'NAD': '15846',
    'ATP': '15422',
    'acetyl-CoA': '15351',
    'B12': '176843',
    'FAD': '16238'
}

QUERIES = {k: f"chebi:{v}" for k, v in CHEBI_MAP.items()}

def fetch_all_uniprots(cofactor, query, max_results=1500, page_size=500):
    """Stáhne všechny UniProt IDs s paginací, logováním a lokální cache."""
    cache_file = os.path.join(structures_dir, f"uniprot_ids_{cofactor}.json")
    
    # 1. Kontrola lokální cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_ids = json.load(f)
        write_log(f"  ⚡ [CACHE] Načteno {len(all_ids)} UniProt ID z lokálního souboru pro {cofactor}.")
        return all_ids[:max_results]
    
    # 2. Stažení z UniProtu
    write_log(f"  🌐 [API] Stahuji UniProt ID z webu...")
    all_ids = []
    offset = 0
    
    while len(all_ids) < max_results:
        encoded_query = quote(query)
        url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_query}&format=json&fields=accession&size={page_size}&offset={offset}"
        
        response = requests.get(url)
        if response.status_code != 200:
            write_log(f"  ❌ Chyba UniProt API: {response.status_code}")
            break
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            write_log("  ℹ️ Konec dat na UniProtu.")
            break
        
        ids = [item['primaryAccession'] for item in results]
        all_ids.extend(ids)
        print(f"    → Dosud staženo: {len(all_ids)}") # Tohle do logu psát nebudeme, jen do konzole
        time.sleep(0.5)
        offset += page_size
    
    # Oříznutí na požadovaný počet
    all_ids = all_ids[:max_results]
    
    # Uložení do cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(all_ids, f)
    write_log(f"  💾 Uloženo {len(all_ids)} záznamů do cache: {cache_file}")
    
    return all_ids

def download_alphafold_structures(cofactor, uniprot_ids, max_downloads=None):
    """Stáhne AlphaFold struktury, kontroluje existenci na disku."""
    if max_downloads is None:
        max_downloads = len(uniprot_ids)
    
    cofactor_dir = os.path.join(structures_dir, cofactor)
    os.makedirs(cofactor_dir, exist_ok=True)
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    for i, uniprot_id in enumerate(uniprot_ids[:max_downloads]):
        # 1. Kontrola, zda už struktura neexistuje na disku
        existing_files = [f for f in os.listdir(cofactor_dir) if uniprot_id in f and f.endswith('.pdb')]
        if existing_files:
            # Tohle vypisujeme jen do konzole (ne do logu), ať nespamujeme
            print(f"  [{i+1}/{max_downloads}] ⚡ PŘESKOČENO (Již existuje): {uniprot_id}")
            skipped += 1
            continue
            
        print(f"  [{i+1}/{max_downloads}] 🌐 STAHOVÁNÍ: {uniprot_id}")
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for fragment in data:
                    pdb_url = fragment.get("pdbUrl")
                    if pdb_url:
                        pdb_response = requests.get(pdb_url, timeout=10)
                        if pdb_response.status_code == 200:
                            filename = os.path.join(cofactor_dir, pdb_url.split("/")[-1])
                            with open(filename, "wb") as f:
                                f.write(pdb_response.content)
                            downloaded += 1
                        else:
                            failed += 1
            elif response.status_code == 404:
                print(f"    ❌ V AlphaFold DB nenalezeno.")
                failed += 1
            else:
                failed += 1
        except Exception as e:
            # Vážnější chyby (např. timeouty) zaznamenáme i do logu
            write_log(f"    ❌ Chyba spojení u {uniprot_id}: {e}")
            failed += 1
        
        time.sleep(0.5) # Ochrana proti banu z EBI
    
    write_log(f"  📊 {cofactor} AF DB report: {downloaded} staženo, {skipped} přeskočeno (cache), {failed} selhalo/nenalezeno.")
    return downloaded, failed, skipped

def merge_fragments(cofactor_dir, uniprot_id):
    """Spojí AlphaFold fragmenty do jednoho proteinu a smaže původní soubory."""
    pdb_files = sorted([f for f in os.listdir(cofactor_dir) if uniprot_id in f and f.endswith('.pdb') and not f.endswith('_MERGED.pdb')])
    
    if not pdb_files:
        return 0 # Není co řešit
        
    merged_filename = os.path.join(cofactor_dir, f"{uniprot_id}_MERGED.pdb")
    if os.path.exists(merged_filename):
        return 0 # Už bylo zpracováno v minulosti
    
    # Pokud je jen jeden soubor, rovnou ho přejmenujeme pro konzistenci dat
    if len(pdb_files) == 1:
        original_file = os.path.join(cofactor_dir, pdb_files[0])
        os.rename(original_file, merged_filename)
        return 1
    
    parser = PDBParser(QUIET=True)
    all_residues = []
    
    for pdb_file in pdb_files:
        filepath = os.path.join(cofactor_dir, pdb_file)
        struct = parser.get_structure(uniprot_id, filepath)
        try:
            chain = struct[0]['A']
            all_residues.extend(list(chain.get_residues()))
        except:
            pass
    
    if len(all_residues) > 0:
        base_structure = parser.get_structure(uniprot_id, os.path.join(cofactor_dir, pdb_files[0]))
        io = PDBIO()
        io.set_structure(base_structure)
        io.save(merged_filename)
        
        # --- ÚKLID: Smazání původních fragmentů ---
        for pdb_file in pdb_files:
            try:
                os.remove(os.path.join(cofactor_dir, pdb_file))
            except OSError as e:
                pass
        return len(pdb_files)
    return 0

# === Hlavní smyčka ===

total_downloaded = 0
total_failed = 0
total_skipped = 0

# Nastavení velikosti datasetu
MAX_UNIPROT_RESULTS = 2000
MAX_ALPHAFOLD_DOWNLOADS = 2000

for cofactor, chebi_query in QUERIES.items():
    write_log(f"\n{'='*50}")
    write_log(f"Kofaktor: {cofactor} (CHEBI: {CHEBI_MAP[cofactor]})")
    write_log(f"{'='*50}")
    
    # 1. KROK: UNIPROT
    uniprots = fetch_all_uniprots(cofactor, chebi_query, max_results=MAX_UNIPROT_RESULTS, page_size=500)
    
    if uniprots:
        write_log(f"\n=== Stažení a kontrola AlphaFold struktur pro {cofactor} ===")
        # 2. KROK: ALPHAFOLD DB
        downloaded, failed, skipped = download_alphafold_structures(cofactor, uniprots, max_downloads=MAX_ALPHAFOLD_DOWNLOADS)
        
        # 3. KROK: MERGE & ÚKLID
        cofactor_dir = os.path.join(structures_dir, cofactor)
        print(f"\n=== Zpracování a úklid fragmentů ===") # Záměrně jen do konzole, ať log není obrovský
        merged_count = 0
        for uid in uniprots:
            merged_count += merge_fragments(cofactor_dir, uid)
            
        write_log(f"  🧩 Zpracováno a sjednoceno do _MERGED.pdb celkem pro {merged_count} struktur.")
        
        total_downloaded += downloaded
        total_failed += failed
        total_skipped += skipped

write_log(f"\n\n{'='*50}")
write_log(f"CELKOVÉ SHRNUTÍ PIPELINE")
write_log(f"{'='*50}")
write_log(f"✅ Nově staženo z AF DB: {total_downloaded}")
write_log(f"⚡ Přeskočeno (již na disku): {total_skipped}")
write_log(f"❌ Selhalo / Nenalezeno v AF DB: {total_failed}")
write_log(f"Konec skriptu.\n")