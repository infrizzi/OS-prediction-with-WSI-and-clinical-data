import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

class VisualDataset(Dataset):
    def __init__(self, split_dir, json_path):
        """
        Args:
            split_dir: split directory path containing .pt files with patch
            json_path: json path containing patient information including OS
        """
        self.split_dir = Path(split_dir)
        # Carichiamo il JSON per recuperare l'OS
        with open(json_path, 'r') as f:
            self.patient_info = json.load(f)
        
        # Lista di tutti i file .pt presenti nello split
        self.file_list = list(self.split_dir.glob("*.pt"))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        patient_id = file_path.stem  # Prende il nome del file senza .pt (es. TCGA-XX-XXXX)
        
        # Caricamento embedding [N, 768]
        features = torch.load(file_path, map_location='cpu')
        
        # Recupero target OS dal JSON
        # Assicurati che il formato del patient_id nel JSON coincida con quello del file
        target_os = self.patient_info[patient_id]['os_months']
        target = torch.tensor([target_os], dtype=torch.float32)
        
        return features, target