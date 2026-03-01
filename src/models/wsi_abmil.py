import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """
    Attention-based MIL aggregator.
    Input:  x [B, N, L] (B=1 in our case, N=number of patches, L=embedding dim)
    Output: M [B, L] aggregated embedding, a [B, N] attention weights
    """
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.2, n_heads=1):
        super().__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.K = n_heads

        self.norm = nn.LayerNorm(self.L)

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(
            nn.Linear(self.D, self.K),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, N, 768]
        x = x.squeeze(0)  # [N, 768]

        x_norm = self.norm(x)  # [N, 768]

        a_v = self.attention_V(x_norm)
        a_u = self.attention_U(x_norm)
        a = self.attention_weights(a_v * a_u)  # [N, K]

        a = a.T                 # [K, N]
        a = F.softmax(a, dim=1) # [K, N]

        M = torch.mm(a, x)          # [K, 768]

        return M, a

class RegressionHead(nn.Module):
    """
    Simple regression head: takes embedding [B, L] -> [B, 1]
    """
    def __init__(self, input_dim=768, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class ABMILRegressor(nn.Module):
    """
    Backward-compatible wrapper:
    - forward(x) returns (pred, attn)
    - forward_features(x) returns (M, attn)
    """
    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.2, n_heads=4):
        super().__init__()
        self.abmil = ABMIL(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, n_heads=n_heads)
        self.head = RegressionHead(input_dim=input_dim, dropout=dropout)

    def forward_features(self, x):
        return self.abmil(x)  # (M, a)

    def forward(self, x):
        M, a = self.abmil(x)
        pred = self.head(M)
        return pred, a