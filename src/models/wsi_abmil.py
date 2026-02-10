import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# ABMIL process is made by following steps:
# 1. For every patch in the bag, we extract the important features 
#    with tanh and computing its relevance with sigmoid
# 2. We multiply the two to get the gated attention scores for every patch
# 3. We normalize the attention scores with softmax to get attention weights
# 4. We compute the bag representation (aggregation) as weighted sum of patch 
#    features
# ============================================================================


# ==========================
# ABMIL MODEL + REGRESSION HEAD
# ==========================

class ABMILRegressor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.K = 1

        # Attention blocks
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.D, self.K)
        )

        # Regression head 
        self.regressor = nn.Sequential(
            nn.Linear(self.L, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    # ==========================
    # FEATURE EXTRACTION
    # ==========================
    def forward_features(self, x):
        """
        Returns:
            M: aggregated WSI embedding [1, L]
            a: attention weights        [1, N]
        """
        # x: [1, N, L]
        x = x.squeeze(0)  # [N, L]

        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        a = self.attention_weights(a_v * a_u)  # [N, 1]

        a = a.T
        a = F.softmax(a, dim=1)

        M = torch.mm(a, x)  # [1, L]
        return M, a

    # ==========================
    # STANDARD FORWARD
    # ==========================
    def forward(self, x):
        M, a = self.forward_features(x)
        pred = self.regressor(M)
        return pred, a