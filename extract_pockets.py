from pathlib import Path
import glob
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
import re


def parse_prediction_file(pfile):
    """Return list of (pocket_id, score, residues_tokens)"""
    pockets = []
    p = Path(pfile)
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # CSV or whitespace
            if ',' in line:
                parts = [x.strip().strip('"').strip("'") for x in line.split(',') if x.strip()]
            else:
                parts = line.split()

            if len(parts) < 2:
                continue
            pid = parts[0]
            # find first numeric token for score
            score = None
            for tok in parts[1:]:
                try:
                    score = float(tok)
                    break
                except ValueError:
                    continue
            if score is None:
                continue
            # residues after score
            # find index of score token
            idx = parts.index(str(int(score))) if str(int(score)) in parts else None
            # fallback: use parts after second token
            residues = parts[2:]
            pockets.append((pid, score, residues))
    return pockets


def parse_residue_token(tok):
    """Try to extract chain and residue number from token like 'A:123', '123', 'A123'"""
    tok = tok.strip()
    # common separators
    tok = tok.replace(';', ',')
    tok = tok.replace('|', ',')
    tok = tok.replace('/', ',')

    # if comma-separated multiple, skip here
    if ',' in tok:
        toks = [t.strip() for t in tok.split(',') if t.strip()]
        out = []
        for t in toks:
            parsed = parse_residue_token(t)
            if parsed:
                out.extend(parsed)
        return out

    m = re.match(r'^(?:(?P<chain>[A-Za-z0-9])[:]?){0,1}(?P<res>\d+)$', tok)
    if m:
        chain = m.group('chain') if m.group('chain') else None
        resseq = int(m.group('res'))
        return [(chain, resseq)]

    # fallback: digits anywhere
    m2 = re.search(r'(\d+)', tok)
    if m2:
        return [(None, int(m2.group(1)))]
    return []


class ResidueSelect(Select):
    def __init__(self, keep_set):
        self.keep_set = keep_set

    def accept_residue(self, residue):
        chain = residue.get_parent().id
        resseq = residue.get_id()[1]
        return (chain, resseq) in self.keep_set or (None, resseq) in self.keep_set


def find_original_pdb(prediction_path):
    """Try to locate original PDB file given prediction csv path.
    Strategy: parent dir name without '_prank_output' is prefix; search in parent.parent for files starting with that prefix and ending with .pdb
    """
    p = Path(prediction_path)
    parent = p.parent.name  # e.g. Q43865_MERGED_prank_output
    prefix = parent.replace('_prank_output', '')
    cofactor_dir = p.parent.parent
    candidates = list(cofactor_dir.glob(f"{prefix}*.pdb"))
    if candidates:
        return candidates[0]
    # fallback: try any pdb in cofactor_dir
    anyp = list(cofactor_dir.glob('*.pdb'))
    return anyp[0] if anyp else None


def extract_pockets_from_prediction(pred_csv, prob_thresh=0.5, out_dir=None):
    pred_csv = Path(pred_csv)
    pockets = parse_prediction_file(pred_csv)
    orig_pdb = find_original_pdb(pred_csv)
    if orig_pdb is None:
        print(f"Original PDB not found for {pred_csv}")
        return 0

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(orig_pdb.stem, str(orig_pdb))

    extracted = 0
    cofactor_dir = pred_csv.parent.parent
    cofactor_name = cofactor_dir.name

    if out_dir is None:
        out_dir = cofactor_dir / f"{cofactor_name}_pockets"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid, score, residues in pockets:
        if score <= prob_thresh:
            continue
        # parse residue tokens into set
        keep = set()
        for tok in residues:
            parsed = parse_residue_token(tok)
            for ch, rs in parsed:
                keep.add((ch, rs))

        keep = expand_keep_set(structure, keep, seq_window=4, ca_cutoff=8.0)

        if not keep:
            continue

        # write PDB with only these residues
        out_file = out_dir / f"{orig_pdb.stem}_pocket_{pid}.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(out_file), ResidueSelect(keep))
        print(f"Wrote pocket: {out_file} (score={score}, residues={len(keep)})")
        extracted += 1

    return extracted


def expand_keep_set(structure, keep_set, seq_window=4, ca_cutoff=8.0):
    expanded = set(keep_set)

    # všechny atomy pro spatial neighborhood
    atoms = list(structure.get_atoms())
    ns = NeighborSearch(atoms)

    for model in structure:
        for chain in model:
            chain_residues = [r for r in chain if r.id[0] == " "]

            for residue in chain_residues:
                chain_id = chain.id
                resseq = residue.id[1]

                if (chain_id, resseq) not in keep_set and (None, resseq) not in keep_set:
                    continue

                # 1) sekvenční okolí v rámci stejného řetězce
                for neighbor in chain_residues:
                    nresseq = neighbor.id[1]
                    if abs(nresseq - resseq) <= seq_window:
                        expanded.add((chain_id, nresseq))

                # 2) prostorové okolí kolem CA atomu
                if residue.has_id("CA"):
                    nearby_residues = ns.search(residue["CA"].coord, ca_cutoff, level="R")
                    for nearby in nearby_residues:
                        nchain = nearby.get_parent().id
                        nresseq = nearby.id[1]
                        expanded.add((nchain, nresseq))

    return expanded


if __name__ == '__main__':
    base = Path(__file__).parent.joinpath('..', 'structures').resolve()
    pattern = "*/**/*_prank_output/*.pdb_predictions.csv"
    files = glob.glob(pattern)
    print(f"found {len(files)} prediction files with pattern: {pattern}")
    total = 0
    for f in files:
        print(f"Processing: {f}")
        try:
            n = extract_pockets_from_prediction(f, prob_thresh=0.5)
            total += n
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"Total pockets extracted: {total}")
