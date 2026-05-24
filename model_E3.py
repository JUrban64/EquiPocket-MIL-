import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from egnn_pytorch import EGNN_Sparse

class GNNBranchE3(nn.Module):
    # Snížen defaultní dropout na 0.1 (10 %)
    def __init__(self, node_dim=1280, hidden_dim=64, num_gnn_layers=3, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        
        self.protein_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                EGNN_Sparse(
                    feats_dim=hidden_dim, 
                    pos_dim=3,
                    m_dim=hidden_dim,
                    update_feats=True,
                    update_coors=True, 
                    dropout=dropout,
                    norm_feats=True,
                    norm_coors=True
                )
            )
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        feats = self.protein_projection(batch.x)
        
        coors = batch.pos
        mean_coors = global_mean_pool(coors, batch.batch)
        coors = coors - mean_coors[batch.batch]
        
        # Sloučení pro lucidrains EGNN
        x_in = torch.cat([coors, feats], dim=-1)
        
        for layer in self.gnn_layers:
            x_in = layer(
                x=x_in, 
                edge_index=batch.edge_index, 
                batch=batch.batch
            )
            
        coors = x_in[:, :3]
        feats = x_in[:, 3:]
            
        feats = self.dropout(feats)
        graph_embs = global_mean_pool(feats, batch.batch)
        
        return graph_embs


class GraphClassifierE3(nn.Module):
    """
    Classifier využívající EGNN_Sparse Encoder (lucidrains).
    Batch musí obsahovat atribut 'pos' s 3D koordidánatami jako float32.
    """
    def __init__(self, node_dim=1280, hidden_dim=64, num_attention_heads=4, dropout=0.1, num_classes=5):
        super().__init__()
        
        # num_attention_heads ignorujeme (není nutné pro global_mean_pool),
        # ale ponecháváme ho v signatuře kvůli kompatibilitě s train.py
        
        self.encoder = GNNBranchE3(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=3,
            dropout=dropout
        )
        
        # Predikční hlava
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def get_embedding(self, batch):
        """
        Vrátí grafový embedding před klasifikační vrstvou.
        """
        # GNNBranchE3 teď přijímá celý batch a rovnou pooluje
        z = self.encoder(batch)
        return z

    def forward(self, batch):
        # 1. Extrakce embeddingu
        z = self.get_embedding(batch)
        # 2. Klasifikace
        return self.classifier(z)
