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
        super(ABMILRegressor, self).__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.K = 1

        # Relevant features extraction (range [-1, 1])
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        # Gating mechanism to avoid non relevant features (range [0, 1])
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        # Single score projection -> attention weights for every patch
        self.attention_weights = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.D, self.K)
            )

        # Regression Branch
        self.regressor = nn.Sequential(
            nn.Linear(self.L, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: [Batch, N, 768] -> [1, N, 768] if we use batch_size=1
        x = x.squeeze(0) # [N, 768]

        # Calcolo pesi attenzione
        a_v = self.attention_V(x) 
        a_u = self.attention_U(x) 
        a = self.attention_weights(a_v * a_u) # [N, 1]
        
        a = torch.transpose(a, 1, 0) # [1, N]
        a = F.softmax(a, dim=1) # Normalized weights

        # Aggregation
        M = torch.mm(a, x) # [1, 768]

        # Regression
        prediction = self.regressor(M) # [1, 1]
        return prediction, a