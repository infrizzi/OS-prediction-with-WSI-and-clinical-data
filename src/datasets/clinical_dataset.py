import torch
from torch.utils.data import Dataset

# ==========================
# TAKE TRAINING DATA
# ==========================

class ClinicalDataset(Dataset):
    def __init__(self, path):
        # Load train bundle
        data = torch.load(path)
        self.features = data['features']
        self.labels = data['labels']
        self.sample_ids = data['sample_ids']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]