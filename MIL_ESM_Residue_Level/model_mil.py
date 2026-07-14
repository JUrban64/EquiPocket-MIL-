import torch
import torch.nn as nn

class AttentionMIL_ESM(nn.Module):
    """
    Attention-based Multi-Instance Learning model directly on ESM embeddings.
    """
    def __init__(self, in_features=1280, hidden_dim=256, num_classes=5, dropout=0.3, num_heads=1, 
                 attention_temp=1.0, gated_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention_temp = attention_temp
        self.gated_attention = gated_attention
        
        mil_in_features = in_features
        
        if gated_attention:
            # Gated Attention podle Ilse et al. (2018)
            self.attention_V = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            )
            self.attention_U = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.Sigmoid()
            )
            self.attention_w = nn.Linear(hidden_dim, num_heads)
        else:
            # Klasická Attention
            self.attention = nn.Sequential(
                nn.Linear(mil_in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, num_heads)  
            )
        
        # Každá hlava vygeneruje svůj vlastní embedding
        classifier_input_dim = mil_in_features * num_heads
        
        # Klasifikátor pro sloučenou reprezentaci (z báglu reziduí nebo kapes)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: [num_instances, in_features] (num_instances = počet kapes nebo počet zbytků celkem)
        
        if self.gated_attention:
            V_out = self.attention_V(x)          # [num_instances, hidden_dim]
            U_out = self.attention_U(x)          # [num_instances, hidden_dim]
            gated_out = V_out * U_out            # [num_instances, hidden_dim]
            A_raw = self.attention_w(gated_out)  # [num_instances, num_heads]
        else:
            A_raw = self.attention(x)            # [num_instances, num_heads]
        
        # Aplikace teplotního škálování před softmaxem (pro zaostření/vyhlazení vah)
        A_scaled = A_raw / self.attention_temp
        
        # Softmax přes dimenzi 0 (instance) pro každou hlavu zvlášť
        A = torch.softmax(A_scaled, dim=0)  # [num_instances, num_heads]
        
        # Seskupení instancí pro každou hlavu zvlášť
        # X: [num_instances, in_features], A.t(): [num_heads, num_instances]
        Z_heads = torch.mm(A.t(), x)        # [num_heads, in_features]
        
        # Zploštění všech hlav do jednoho vektoru
        Z = Z_heads.view(1, -1)             # [1, num_heads * in_features]
        
        # Klasifikace celého proteinu
        logits = self.classifier(Z)         # [1, num_classes]
        return logits, A
