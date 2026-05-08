import pandas as pd 
import subprocess
import os 
from pathlib import Path



def prank_predict(structure_file, output_prefix, threads=4):
    """Spustí PRANK pro daný PDB soubor a uloží výsledky."""
    cmd = [
        "p2rank_2.5.1/prank", "predict",
        "-c", "alphafold",
        "-f", str(structure_file),
        "-o", str(output_prefix),
        "-t", str(threads),
    ]
    print(f"Spouštím PRANK: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"PRANK dokončen: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Chyba při spouštění PRANK pro {structure_file}:")
        if e.stderr:
            print(e.stderr.strip())
        return False
    except FileNotFoundError:
        print("Chyba: přikaz prank nebyl nalezen v PATH.")
        return False



if __name__ == "__main__":
    structures_root = Path("./structures")
    threads = 4

    if not structures_root.exists():
        raise FileNotFoundError(f"Složka neexistuje: {structures_root.resolve()}")

    # Rekurzivně najde všechny PDB i v podsložkách (NAD, ATP, FAD, ...)
    pdb_files = sorted(structures_root.rglob("*.pdb"))
    print(f"Nalezeno PDB souborů: {len(pdb_files)}")

    ok = 0
    fail = 0
    skipped = 0

    for pdb_path in pdb_files:
        output_prefix = str(pdb_path.with_suffix("")) + "_prank_output"

        # Preskoci uz zpracovane soubory
        if Path(output_prefix + ".csv").exists():
            print(f"Preskakuji (uz existuje): {output_prefix}.csv")
            skipped += 1
            continue

        success = prank_predict(pdb_path, output_prefix, threads=threads)
        if success:
            ok += 1
        else:
            fail += 1

    print("\nSouhrn:")
    print(f"OK: {ok}")
    print(f"Failed: {fail}")
    print(f"Skipped: {skipped}")