import torch.nn as nn

# ==========================
# BUILD THE MODEL
# ==========================

class ClinicalMLP(nn.Module):
    def __init__(self, input_dim=463, embedding_dim=64):
        super().__init__()
        
        self.embedding_extractor = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final regression head
        self.regression_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        emb = self.embedding_extractor(x)
        return self.regression_head(emb)