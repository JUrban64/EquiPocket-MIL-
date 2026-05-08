from torch_geometric.data import Data
import torch
import numpy as np
import argparse
import json
import os
import importlib.util
from esm2_feature_ex import ESMFeatureExtractor


def create_node_features(bs_info, **kwargs):
    """
    Extrakce node features pro proteinová residua.
    Pro náš EGNN model spoléháme čistě na ESM embeddingy.
    """
    if 'esm_embeddings' in bs_info:
        return np.asarray(bs_info['esm_embeddings'], dtype=np.float32)
    
    n = bs_info.get('n_binding_site', 0)
    # Defaultní ESM dimenze je 1280 (esm2_t33_650M_UR50D)
    return np.zeros((n, 1280), dtype=np.float32)



# Typy hran
EDGE_TYPE_PP = 0   # Protein–Protein  (kontaktní mapa)

# Typy uzlů
NODE_TYPE_PROTEIN = 0

# Typy interakcí (pro edge feature encoding)
INTERACTION_TYPES = ['hbond_candidate', 'hydrophobic', 'ionic', 'other']
ITYPE_TO_IDX = {t: i for i, t in enumerate(INTERACTION_TYPES)}

# Pevná dimenze edge features: [distance_norm, hbond, hydrophobic, ionic, other]
EDGE_ATTR_DIM = 5


class BindingSiteGraphDataset:
    """
    Dataset of protein-ligand interaction graphs.
    
    Každý graf obsahuje:
      - Protein uzly (residues v binding site) s proteinovými features
      - Ligand uzly (atomy kofaktoru) s ligandovými features
      - Tři typy hran:
          P-P: kontaktní mapa proteinových residues
          P-L: protein-ligand interakce (distance-based)
          L-L: kovalentní vazby uvnitř ligandu
      - node_type: [N] tensor (0=protein, 1=ligand)
      - edge_type: [E] tensor (0=PP, 1=PL, 2=LL)
      - cofactor_id: str ('NAD', 'FAD', ...)
    """
    
    def __init__(self, binding_sites_data, feature_config=None):
        """
        Args:
            binding_sites_data: list of binding site info dicts
            feature_config: dict specifying which protein features to use
        """
        self.data = binding_sites_data
        
        if feature_config is None:
            feature_config = {
                'use_esm': True,
            }
        
        self.feature_config = feature_config
        # Labely a sekvence se drží zvlášť pro split bez stavby grafů
        self.labels = []
        valid_indices = []
        for i, bs in enumerate(self.data):
            lbl = int(bs.get('label', -1))
            if lbl == -1:
                lig = bs.get('ligand_name', '')
                act = bs.get('actual_ligand_name', '')
                try:
                    # Provide an error fallback lookup in case 'label' missing
                    from Binding_site_ex import COFACTOR_FUNCTIONAL_GROUPS
                    supported = list(COFACTOR_FUNCTIONAL_GROUPS.keys())
                    if act in supported:
                        lbl = supported.index(act)
                except Exception:
                    pass
            if lbl != -1:
                self.labels.append(lbl)
                valid_indices.append(i)
                
        # Filter data to only valid labelled samples
        self.data = [self.data[i] for i in valid_indices]
        self.sequences = [bs.get('binding_site_sequence', '') for bs in self.data]
        self.pdb_ids = [bs.get('pdb_file', '') for bs in self.data]

    @staticmethod
    def load_binding_sites_json(input_json):
        """
        Načte JSON z Binding_site_ex.py (save_binding_sites) a vrátí list záznamů.

        Podporuje formát:
            - dict: {protein_id: bs_info}
            - list: [bs_info, ...]
        """
        with open(input_json, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            data = []
            for protein_id, bs_info in raw.items():
                item = dict(bs_info)
                item.setdefault('protein_id', protein_id)
                data.append(item)
            return data

        if isinstance(raw, list):
            return raw

        raise ValueError(
            f"Unsupported JSON format in {input_json}: expected dict or list"
        )

    def build_graphs(self):
        """Postavi vsechny grafy a vrati je jako list PyG Data objektu."""
        return [self[i] for i in range(len(self))]

    def save_graphs(self, output_path, graphs=None):
        """
        Uloží grafy do .pt souboru.

        Pokud `graphs` není předán, grafy se sestaví z aktuálního datasetu.
        """
        if graphs is None:
            graphs = self.build_graphs()

        payload = {
            'graphs': graphs,
            'feature_config': self.feature_config,
            'num_graphs': len(graphs),
        }
        torch.save(payload, output_path)

    @staticmethod
    def load_graphs(input_path, map_location='cpu'):
        """Načte uložené grafy z .pt souboru."""
        payload = torch.load(input_path, map_location=map_location)
        if isinstance(payload, dict) and 'graphs' in payload:
            return payload
        return {'graphs': payload}
    
    def _build_single_graph(self, bs_info):
        """
        Build PyG graph for single binding site.
        
        Protein-ligand interaction graph:
            Nodes:  [0 .. n_prot-1] = protein residues
                    [n_prot .. n_prot+n_lig-1] = ligand atoms
            Edges:  P-P (contact map), P-L (interactions), L-L (bonds)
        
        Returns:
            PyG Data object with extra attributes:
                - node_type: [N] int tensor
                - edge_type: [E] int tensor
                - edge_interaction: [E, 4] float tensor (interaction type one-hot, only for P-L edges)
                - n_protein_nodes: int
                - n_ligand_nodes: int
                - cofactor_id: str
        """
        # ---- Protein node features ----
        protein_features = create_node_features(
            bs_info, **self.feature_config
        )
        n_prot = protein_features.shape[0]
        protein_dim = protein_features.shape[1]
        
        x = torch.FloatTensor(protein_features)
        
        # ---- Node type ----
        node_type = torch.zeros(n_prot, dtype=torch.long)

        # ---- POS (3D coordinates for E(3) equivariant models) ----
        protein_coords = bs_info.get('protein_coords', [])
        if len(protein_coords) == n_prot:
            pos_prot = np.array(protein_coords, dtype=np.float32)
        else:
            pos_prot = np.zeros((n_prot, 3), dtype=np.float32)
            
        pos_tensor = torch.tensor(pos_prot, dtype=torch.float32)
        
        # ---- EDGES ----
        # Všechny edge_attr mají PEVNOU dimenzi EDGE_ATTR_DIM = 5:
        #   [distance_norm, hbond, hydrophobic, ionic, other]
        all_edges = []      # list of [src, dst]
        all_edge_types = [] # list of int
        all_edge_attr = []  # list of [EDGE_ATTR_DIM] float vectors
        
        # 1) P-P edges: z kontaktní mapy (vektorizovaně přes np.where)
        # JSON z Binding_site_ex.py obsahuje contact_map jako list of lists,
        # proto sjednotíme na numpy array.
        contact_map = np.asarray(bs_info['contact_map'], dtype=np.float32)
        pp_rows, pp_cols = np.where(contact_map > 0.5)
        pp_mask = pp_rows != pp_cols
        pp_rows = pp_rows[pp_mask]
        pp_cols = pp_cols[pp_mask]
        if len(pp_rows) > 0:
            pp_weights = contact_map[pp_rows, pp_cols]
            pp_edges = np.stack([pp_rows, pp_cols], axis=1)          # [E_pp, 2]
            pp_attr = np.zeros((len(pp_rows), EDGE_ATTR_DIM))       # [E_pp, 5]
            pp_attr[:, 0] = pp_weights
            all_edges.extend(pp_edges.tolist())
            all_edge_types.extend([EDGE_TYPE_PP] * len(pp_rows))
            all_edge_attr.extend(pp_attr.tolist())
        
        # ---- Sloučení hran ----
        if all_edges:
            edge_index = torch.LongTensor(all_edges).t().contiguous()
            edge_type = torch.LongTensor(all_edge_types)
            edge_attr = torch.FloatTensor(all_edge_attr)  # [E, 5] – vždy stejná dim
        else:
            # Fallback: fully connected protein-only
            edge_list = [[i, j] for i in range(n_prot) for j in range(n_prot) if i != j]
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_type = torch.zeros(len(edge_list), dtype=torch.long)
            edge_attr = torch.zeros(len(edge_list), EDGE_ATTR_DIM)
        
        # ---- Label ----
        label_val = int(bs_info.get('label', -1))
        # (Filtered above, so this should not be -1 in practice)
        y = torch.LongTensor([label_val])
        
        # ---- Sestavení PyG Data ----
        graph = Data(
            x=x,
            pos=pos_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            node_type=node_type,
            y=y,
            # Metadata
            sequence=bs_info['binding_site_sequence'],
            full_sequence=bs_info.get('full_sequence', bs_info['binding_site_sequence']),
            pdb_id=bs_info['pdb_file'],
            n_residues=bs_info['n_binding_site'],
            n_protein_nodes=n_prot,
            protein_dim=protein_dim,
            cofactor_id=bs_info.get('ligand_name', 'UNK'),
            pocket_id=bs_info.get('pocket_id', ''),
        )
        
        return graph
    
    def _contact_map_to_edges(self, contact_map, threshold=0.5):
        """
        Convert contact map to edge list (P-P edges only).
        Vektorizovaná verze pomocí np.where.
        
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1] (contact probability)
        """
        rows, cols = np.where(contact_map > threshold)
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        
        if len(rows) == 0:
            # Fallback: fully connected
            n = contact_map.shape[0]
            rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            rows, cols = rows.ravel(), cols.ravel()
            mask = rows != cols
            rows, cols = rows[mask], cols[mask]
            edge_weights = np.ones(len(rows), dtype=np.float32)
        else:
            edge_weights = contact_map[rows, cols].astype(np.float32)
        
        edge_index = torch.LongTensor(np.stack([rows, cols])).contiguous()
        edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Lazy: sestaví graf až když ho DataLoader potřebuje."""
        graph = self._build_single_graph(self.data[idx])
        graph.y = torch.LongTensor([self.labels[idx]])
        return graph


def _attach_esm_embeddings_batch(batch_items, esm_extractor):
    """Compute ESM embeddings on-the-fly for one batch of binding-site records."""
    def _resolve_esm_dim(local_embeddings):
        for emb in local_embeddings:
            if emb is not None and hasattr(emb, 'shape') and len(emb.shape) == 2:
                return int(emb.shape[1])
        if hasattr(esm_extractor, 'model') and hasattr(esm_extractor.model, 'config'):
            hidden_size = getattr(esm_extractor.model.config, 'hidden_size', None)
            if hidden_size is not None:
                return int(hidden_size)
        return 1280

    full_sequences = [item.get('full_sequence', '') for item in batch_items]
    try:
        embeddings_batch = esm_extractor.batch_extract(
            full_sequences, batch_size=len(full_sequences)
        )
    except Exception as e:
        print(f"Batch ESM extraction failed, switching to per-item mode: {e}")
        embeddings_batch = []
        for i, seq in enumerate(full_sequences):
            try:
                embeddings_batch.append(esm_extractor.extract_embeddings(seq))
            except Exception as item_e:
                print(
                    f"Error extracting ESM for item {i} "
                    f"(protein_id={batch_items[i].get('protein_id', batch_items[i].get('pdb_file', 'unknown'))}): {item_e}"
                )
                embeddings_batch.append(None)

    esm_dim = _resolve_esm_dim(embeddings_batch)

    for i, item in enumerate(batch_items):

        try:
            bs_indices = item.get('binding_site_indices', [])
            n_bs = int(item.get('n_binding_site', 0))
            full_emb = embeddings_batch[i]
            if full_emb is None:
                raise ValueError('ESM embedding is None')

            if not bs_indices:
                raise ValueError(
                    f"No binding_site_indices for protein {item.get('protein_id', item.get('pdb_file', 'unknown'))}"
                )

            valid_indices = [idx for idx in bs_indices if idx < full_emb.shape[0]]
            if not valid_indices:
                raise ValueError(
                    f"No valid BS indices after ESM truncation for protein {item.get('protein_id', item.get('pdb_file', 'unknown'))}"
                )

            bs_emb = full_emb[np.asarray(valid_indices, dtype=np.int64), :]
        except Exception as e:
            print(f"Error extracting ESM embeddings for item {i} (protein_id={item.get('protein_id', item.get('pdb_file', 'unknown'))}): {e}")
            bs_emb = np.zeros((n_bs, esm_dim), dtype=np.float32)
            valid_indices = []

        if bs_emb.shape[0] != n_bs:
            # Keep graph dimensions consistent after truncation.
            original_bs_indices = list(bs_indices)
            item['binding_site_indices'] = valid_indices
            item['n_binding_site'] = len(valid_indices)
            valid_set = set(valid_indices)
            seq_chars = list(item.get('binding_site_sequence', ''))
            keep_pos = [k for k, idx in enumerate(original_bs_indices) if idx in valid_set]
            if seq_chars:
                item['binding_site_sequence'] = ''.join(seq_chars[k] for k in keep_pos if k < len(seq_chars))
            cm = np.asarray(item.get('contact_map', []), dtype=np.float32)
            if cm.size > 0:
                idx_arr = np.asarray(keep_pos, dtype=np.int64)
                item['contact_map'] = cm[np.ix_(idx_arr, idx_arr)].tolist()

        item['esm_embeddings'] = np.asarray(bs_emb, dtype=np.float32)


def build_and_save_graphs_batched(binding_sites, output_dir, graph_batch_size=8,
                                  esm_model_name='facebook/esm2_t33_650M_UR50D'):
    """Build graphs on-the-fly in batches and save each batch separately."""
    os.makedirs(output_dir, exist_ok=True)

    esm_extractor = ESMFeatureExtractor(model_name=esm_model_name)
    total = len(binding_sites)
    batch_files = []
    total_saved = 0
    total_skipped = 0

    for start in range(0, total, graph_batch_size):
        end = min(start + graph_batch_size, total)
        batch_items = [dict(binding_sites[i]) for i in range(start, end)]

        _attach_esm_embeddings_batch(batch_items, esm_extractor)

        dataset = BindingSiteGraphDataset(
            batch_items,
            feature_config={
                'use_esm': True,
                'use_blosum': False,
                'use_physchem': False,
                'use_position': False,
            }
        )

        graphs = []
        skipped_in_batch = 0
        for local_idx in range(len(dataset)):
            try:
                graphs.append(dataset[local_idx])
            except Exception as e:
                skipped_in_batch += 1
                total_skipped += 1
                item = batch_items[local_idx]
                print(
                    f"Skipping graph for item {start + local_idx} "
                    f"(protein_id={item.get('protein_id', item.get('pdb_file', 'unknown'))}): {e}"
                )

        if not graphs:
            batch_idx = start // graph_batch_size
            print(f"Skipping batch {batch_idx} ({start}:{end}) - no valid graphs")
            del dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        batch_idx = start // graph_batch_size
        batch_path = os.path.join(output_dir, f'graphs_batch_{batch_idx:05d}.pt')
        torch.save(
            {
                'graphs': graphs,
                'batch_index': batch_idx,
                'start': start,
                'end': end,
                'num_graphs': len(graphs),
                'num_skipped': skipped_in_batch,
                'feature_config': dataset.feature_config,
                'esm_model_name': esm_model_name,
            },
            batch_path,
        )

        batch_files.append(os.path.basename(batch_path))
        total_saved += len(graphs)

        del graphs
        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Saved batch {batch_idx} ({start}:{end}) -> {batch_path}")

    manifest = {
        'num_batches': len(batch_files),
        'num_graphs': total_saved,
        'num_skipped': total_skipped,
        'graph_batch_size': graph_batch_size,
        'esm_model_name': esm_model_name,
        'files': batch_files,
    }
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path, total_saved


# Create dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build and save protein-ligand graphs from binding site JSON.'
    )
    parser.add_argument(
        '--input-json',
        default='binding_sites_by_protein.json',
        help='Path to JSON produced by Binding_site_ex.py'
    )
    parser.add_argument(
        '--output-dir',
        default='binding_site_graph_batches',
        help='Output directory for serialized graph batches'
    )
    parser.add_argument(
        '--graph-batch-size',
        type=int,
        default=8,
        help='How many structures to process/save per batch'
    )
    parser.add_argument(
        '--esm-model-name',
        default='facebook/esm2_t33_650M_UR50D',
        help='HuggingFace ESM model name for on-the-fly embeddings'
    )
    parser.add_argument(
        '--split-file',
        default=None,
        help='Optional path to text file containing protein IDs to filter from the input JSON (e.g. train.txt).'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    binding_sites = BindingSiteGraphDataset.load_binding_sites_json(args.input_json)
    
    if args.split_file:
        print(f"Filtering dataset using IDs from: {args.split_file}")
        with open(args.split_file, 'r', encoding='utf-8') as f:
            allowed_ids = set(line.strip() for line in f if line.strip())
            
        filtered_sites = [bs for bs in binding_sites if bs.get('protein_id') in allowed_ids]
        print(f"Filtered down to {len(filtered_sites)} binding sites from {len(binding_sites)}")
        binding_sites = filtered_sites

    manifest_path, total_graphs = build_and_save_graphs_batched(
        binding_sites,
        output_dir=args.output_dir,
        graph_batch_size=args.graph_batch_size,
        esm_model_name=args.esm_model_name,
    )

    print(f"Created {total_graphs} graphs")
    print(f"Saved graph batches to: {args.output_dir}")
    print(f"Manifest: {manifest_path}")