import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from egnn_pytorch import EGNN_Sparse

class GNNBranchE3(nn.Module):
    """
    E(n)-Equivariant GNN využívající egnn-pytorch (lucidrains).
    Tento model zachovává rotační a translační symetrie pro 3D souřadnice.
    """
    def __init__(self, node_dim=1280, hidden_dim=64, num_gnn_layers=3, dropout=0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        
        # Lineární projekce původních vlastností uzlů (např. ESM)
        self.protein_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Samotné EGNN Sparse vrstvy.
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                EGNN_Sparse(
                    feats_dim=hidden_dim, 
                    pos_dim=3,
                    m_dim=hidden_dim,
                    update_feats=True,
                    update_coors=True,
                    dropout=dropout
                )
            )
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        # 1. Projekce původních ESM features
        x_feats = self.protein_projection(batch.x)
        
        # 2. Centrování souřadnic (ZABRÁNÍ EXPLOZI LOSS A NAN HODNOTÁM)
        coors = batch.pos
        # Spočítáme těžiště pro každý graf v batchi a odečteme ho
        mean_coors = global_mean_pool(coors, batch.batch)
        coors = coors - mean_coors[batch.batch]
        
        feats = x_feats
        
        for layer in self.gnn_layers:
            # Sbalíme pos a feats pro layer
            x_in = torch.cat([coors, feats], dim=-1)
            
            # EGNN_Sparse vrací zřetězené [coors_out, feats_out]
            out = layer(x=x_in, edge_index=batch.edge_index, batch=batch.batch)
            
            # Rozbalíme zpět
            coors = out[:, :3]
            feats = out[:, 3:]
            
        feats = self.dropout(feats)
        
        # 3. Pooling pro celý graf
        # Získáme 1 vektor pro každý graf v batchi.
        graph_embs = global_mean_pool(feats, batch.batch)
        
        return graph_embs


class GraphClassifierE3(nn.Module):
    """
    Classifier využívající EGNN_Sparse Encoder (lucidrains).
    Batch musí obsahovat atribut 'pos' s 3D koordidánatami jako float32.
    """
    def __init__(self, node_dim=1280, hidden_dim=64, num_attention_heads=4, dropout=0.5, num_classes=2):
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
