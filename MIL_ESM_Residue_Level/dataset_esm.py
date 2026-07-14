import torch
import numpy as np
import os
import json
from collections import defaultdict
import argparse
from tqdm import tqdm

def load_data_from_pyg_batches(manifest_path, mode='pockets'):
    """
    Načte ESM embeddingy z PyG grafů a zkompletuje je do struktury pro MIL.
    mode: 'pockets' (zprůměruje rezidua do kapes) nebo 'residues' (nechá všechna rezidua).
    """
    print(f"Načítám ESM embeddingy (mód: {mode}) podle {manifest_path}...")
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    # Ukládáme si pro každý protein seznam embeddingů
    # Pokud mode == 'pockets', bude to seznam vektorů 1280 (1 per kapsa)
    # Pokud mode == 'residues', bude to seznam matic [num_residues, 1280], které pak slepíme.
    bags_dict = defaultdict(list)
    labels_dict = {}
    
    for batch_file in tqdm(manifest['files'], desc="Načítání grafů"):
        full_path = os.path.join(base_dir, batch_file)
        if not os.path.exists(full_path):
            continue
            
        data_batch = torch.load(full_path, map_location='cpu', weights_only=False)
        graphs = data_batch.get('graphs', [])
        
        for g in graphs:
            if g.x is None or g.x.shape[0] == 0:
                continue
                
            label = g.y.item() if hasattr(g.y, 'item') else int(g.y[0])
            
            # Získání Protein ID
            pocket_id = getattr(g, 'pocket_id', '')
            if pocket_id:
                protein_id = pocket_id.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            else:
                pdb_id = getattr(g, 'pdb_id', '')
                protein_id = pdb_id.replace('.pdb', '')
                
            labels_dict[protein_id] = label
            
            # g.x je tensor [num_residues, 1280]
            if mode == 'pockets':
                # Zprůměrujeme uzly -> 1 vektor pro kapsu
                pocket_emb = g.x.mean(dim=0).unsqueeze(0) # [1, 1280]
                bags_dict[protein_id].append(pocket_emb)
            elif mode == 'residues':
                # Přidáme všechna rezidua z této kapsy
                bags_dict[protein_id].append(g.x)
            else:
                raise ValueError(f"Neznámý mód: {mode}")

    bag_list = []
    total_instances = 0
    
    for pid, features_list in bags_dict.items():
        # features_list je seznam tensorů. Sloučíme je přes dim=0 (dimenze instancí)
        bag_features = torch.cat(features_list, dim=0) # [num_instances, 1280]
        total_instances += bag_features.shape[0]
        
        bag_list.append({
            'protein_id': pid,
            'features': bag_features,
            'label': torch.LongTensor([labels_dict[pid]])
        })
        
    print(f"\nCelkem proteinů (bags): {len(bag_list)}")
    print(f"Celkem instancí (kapsy nebo rezidua celkem): {total_instances}")
    if len(bag_list) > 0:
        print(f"Průměrně instancí na bag: {total_instances / len(bag_list):.1f}")
        
    return bag_list

def load_split_ids(base_dir):
    train_ids, val_ids, test_ids = set(), set(), set()
    train_path = os.path.join(base_dir, 'train_mil.txt')
    val_path = os.path.join(base_dir, 'validation_mil.txt')
    test_path = os.path.join(base_dir, 'test_mil.txt')
    
    if os.path.exists(train_path):
        train_ids = set(open(train_path).read().splitlines())
    if os.path.exists(val_path):
        val_ids = set(open(val_path).read().splitlines())
    if os.path.exists(test_path):
        test_ids = set(open(test_path).read().splitlines())
    return train_ids, val_ids, test_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-path', default='../p2rank_graph_batches/manifest.json')
    parser.add_argument('--mode', choices=['pockets', 'residues'], default='residues')
    args = parser.parse_args()
    bags = load_data_from_pyg_batches(args.manifest_path, mode=args.mode)
