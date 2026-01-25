import torch.nn as nn

class ClinicalMLP(nn.Module):
    def __init__(self, input_dim=463, embedding_dim=64):
        super().__init__()
        
        self.embedding_extractor = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Layer 2
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Head di regressione lineare (nessuna attivazione finale)
        self.regression_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        emb = self.embedding_extractor(x)
        return self.regression_head(emb)