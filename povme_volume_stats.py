import os
import argparse
import pandas as pd
import subprocess
import glob
import tempfile
import shutil
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate POVME pocket volumes for P2Rank predictions")
    parser.add_argument("--structures-dir", default="./structures", help="Directory with PDBs and *_prank_output")
    parser.add_argument("--radius", type=float, default=15.0, help="PointsInclusionSphere radius (default: 15.0 A)")
    parser.add_argument("--prob", type=float, default=0.8, help="Minimum probability of pocket to be measured (default: 0.8)")
    parser.add_argument("--povme-bin", default="../POVME/povme", help="Command to execute POVME (e.g. POVME3.py or POVME)")
    parser.add_argument("--out-csv", default="pocket_volumes_stats.csv", help="Output file for volumes")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N proteins (for testing)")
    return parser.parse_args()

def run_povme(povme_bin, pdb_file, center_x, center_y, center_z, radius, output_prefix):
    # Dynamically create the POVME .ini file
    # We use a small box around the center for the contiguous pocket seed
    seed_r = 2.0
    ini_content = f"""
PDBFileName                 {pdb_file}
GridSpacing                 1.0
PointsInclusionSphere       {center_x} {center_y} {center_z} {radius}
ContiguousPocketSeedBox     {center_x - seed_r} {center_x + seed_r} {center_y - seed_r} {center_y + seed_r} {center_z - seed_r} {center_z + seed_r}
SavePoints                  false
OutputFilenamePrefix        {output_prefix}
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(ini_content)
        ini_path = f.name
        
    try:
        # Run POVME
        subprocess.run([povme_bin, ini_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Read the resulting volume
        vol_file = f"{output_prefix}volumes.tabbed.txt"
        volume = np.nan
        if os.path.exists(vol_file):
            with open(vol_file, 'r') as vf:
                lines = vf.readlines()
                if len(lines) > 1:
                    # Format is usually: Frame Volume
                    # We just take the last element of the second line
                    parts = lines[1].strip().split()
                    if parts:
                        volume = float(parts[-1])
                        
        return volume
    except Exception as e:
        print(f"Error running POVME for {pdb_file}: {e}")
        return np.nan
    finally:
        if os.path.exists(ini_path):
            os.unlink(ini_path)
        # Clean up povme generated files (we just need the volume value)
        for p_file in glob.glob(f"{output_prefix}*"):
            try:
                os.unlink(p_file)
            except:
                pass

def main():
    args = parse_args()
    
    structures_dir = Path(args.structures_dir)
    if not structures_dir.exists():
        print(f"Directory {structures_dir} does not exist.")
        return
        
    # Find all P2Rank prediction CSVs
    csv_files = list(structures_dir.rglob("*_predictions.csv"))
    if args.limit:
        csv_files = csv_files[:args.limit]
        print(f"Limiting to {args.limit} proteins for testing.")
        
    print(f"Found {len(csv_files)} P2Rank prediction files.")
    
    # We will assume folder structure: structures/<class_name>/<pdb_name>_prank_output/...
    # Or at least that the cofactor class can be extracted from the path.
    supported_classes = ['ATP', 'NAD', 'FAD', 'acetyl-CoA', 'B12']
    
    results = []
    
    temp_povme_dir = tempfile.mkdtemp(prefix="povme_tmp_")
    
    try:
        for csv_path in csv_files:
            # Figure out class
            cofactor_class = "UNKNOWN"
            for c in supported_classes:
                if c in csv_path.parts:
                    cofactor_class = c
                    break
                    
            # The original PDB should be in the parent dir of the prank_output dir
            pdb_name = csv_path.name.replace("_predictions.csv", "")
            prank_output_dir = csv_path.parent
            pdb_file = prank_output_dir.parent / pdb_name
            
            if not pdb_file.exists():
                # Maybe it doesn't end with .pdb exactly in the folder name, try fallback
                fallback = prank_output_dir.parent / f"{pdb_name}.pdb"
                if fallback.exists():
                    pdb_file = fallback
                else:
                    print(f"PDB file not found for {csv_path}, skipping.")
                    continue
                    
            try:
                # Read CSV, P2Rank CSV uses spaces, so we use skipinitialspace
                df = pd.read_csv(csv_path, skipinitialspace=True)
                # Ensure column names don't have spaces
                df.columns = [c.strip() for c in df.columns]
                
                # Filter by probability >= threshold
                valid_pockets = df[df['probability'] >= args.prob]
                
                for _, row in valid_pockets.iterrows():
                    pocket_name = row['name']
                    cx, cy, cz = row['center_x'], row['center_y'], row['center_z']
                    prob = row['probability']
                    
                    # Run POVME
                    out_prefix = os.path.join(temp_povme_dir, f"{pdb_name}_{pocket_name}_")
                    vol = run_povme(args.povme_bin, str(pdb_file), cx, cy, cz, args.radius, out_prefix)
                    
                    results.append({
                        'protein_id': pdb_name,
                        'cofactor': cofactor_class,
                        'pocket_name': pocket_name,
                        'probability': prob,
                        'volume': vol
                    })
                    print(f"Processed {pdb_name} - {pocket_name}: {vol} A^3")
                    
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
                
    finally:
        shutil.rmtree(temp_povme_dir, ignore_errors=True)
        
    if not results:
        print("No pockets met the criteria or could be measured.")
        return
        
    df_results = pd.DataFrame(results)
    df_results = df_results.dropna(subset=['volume'])
    
    df_results.to_csv(args.out_csv, index=False)
    print(f"\\nSaved raw volumes to {args.out_csv}")
    
    print("\\n=== Pocket Volume Statistics by Cofactor ===")
    stats = df_results.groupby('cofactor')['volume'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print(stats.to_string())

if __name__ == "__main__":
    main()
