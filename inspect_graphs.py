import torch

try:
    payload = torch.load('gt_graph_batches_train/graphs_batch_0.pt', map_location='cpu', weights_only=False)
    graphs = payload['graphs'] if isinstance(payload, dict) and 'graphs' in payload else payload
    print("Found", len(graphs), "graphs")
    
    for i, g in enumerate(graphs[:5]):
        pid = getattr(g, 'protein_id', getattr(g, 'pdb_id', getattr(g, 'pocket_id', '')))
        cleaned = pid.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
        print(f"Graph {i}: original={pid}, cleaned={cleaned}")
except Exception as e:
    print("Error:", e)
