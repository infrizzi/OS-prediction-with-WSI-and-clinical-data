import torch
import torch.nn as nn

class TransMIL(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, dropout=0.2):
        super(TransMIL, self).__init__()
        
        # 1. Initial linear projection to match the Transformer's latent dimension
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. CLS token initialization (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 3. Transformer Layers initialization (2 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=1, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: [B, N, 768]
        """
        B, N, _ = x.shape
        
        # Linear projection to hidden_dim
        h = self.fc(x) # [B, N, hidden_dim]
        
        # CLS token insertion
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1) # [B, N+1, hidden_dim]
        
        # Self-Attention between all patches + CLS token
        h = self.transformer(h) # [B, N+1, hidden_dim]
        
        # Extracting CLS only for regression -> global representation of the bag
        vis_emb = self.norm(h[:, 0]) # [B, hidden_dim]
        
        # Return
        dummy_attn = torch.zeros(B, N, device=x.device)
        return vis_emb, dummy_attn
    
class RegressionHead(nn.Module):
    """
    Simple regression head: takes embedding [B, L] -> [B, 1]
    """
    def __init__(self, input_dim=768, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class TransMILRegressor(nn.Module):
    """
    Backward-compatible wrapper:
    - forward(x) returns (pred, attn)
    - forward_features(x) returns (M, attn)
    """
    def __init__(self, input_dim=768, hidden_dim=768, dropout=0.2):
        super().__init__()
        self.transmil = TransMIL(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = RegressionHead(input_dim=hidden_dim, dropout=dropout)

    def forward_features(self, x):
        return self.transmil(x)  # (M, a)

    def forward(self, x):
        M, a = self.transmil(x)
        pred = self.head(M)
        return pred, a