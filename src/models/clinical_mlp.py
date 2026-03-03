import torch.nn as nn


class ClinicalEncoder(nn.Module):
    """
    Clinical feature encoder.
    Input:  x [B, input_dim]
    Output: emb [B, embedding_dim]
    """
    def __init__(self, input_dim=50, embedding_dim=463):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 463),
            nn.LayerNorm(463),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(463, 463),
            nn.LayerNorm(463),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(463, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)
    
class ClinicalRegressionHead(nn.Module):
    """
    Regression head for clinical embeddings.
    Input:  emb [B, embedding_dim]
    Output: pred [B, 1]
    """
    def __init__(self, embedding_dim=463):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.fc(x)
    
class ClinicalMLP(nn.Module):
    """
    Backward-compatible wrapper.
    - forward(x)        -> prediction (unimodal clinical)
    - forward_features  -> embedding (for MCAT)
    """
    def __init__(self, input_dim=463, embedding_dim=463):
        super().__init__()
        self.encoder = ClinicalEncoder(input_dim, embedding_dim)
        self.head = ClinicalRegressionHead(embedding_dim)

    # ==========================
    # FEATURE EXTRACTION
    # ==========================
    def forward_features(self, x):
        return self.encoder(x)

    # ==========================
    # STANDARD FORWARD
    # ==========================
    def forward(self, x):
        emb = self.encoder(x)
        return self.head(emb)