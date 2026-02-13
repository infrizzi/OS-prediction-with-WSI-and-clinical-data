import torch
from torch.utils.data import Dataset
from pathlib import Path

class MCATDataset(Dataset):
    def __init__(self, clinical_split_path, visual_dir):
        """
        Args:
            clinical_split_path (str/Path): Percorso al file .pt clinico (es. train_data.pt)
            visual_dir (str/Path): Directory dello split visivo corrispondente (es. .../visual_splits/train)
                                   Contiene i file già aggregati nominati come 'PATIENT_ID.pt'
        """
        # 1. Carichiamo il bundle clinico
        # Struttura: {'features': tensor, 'labels': tensor, 'sample_ids': list}
        self.clinical_bundle = torch.load(clinical_split_path)
        self.visual_dir = Path(visual_dir)
        
        # 2. Prepariamo gli indici validi
        # Filtriamo i pazienti clinici per assicurarci che esista il corrispettivo visivo
        self.valid_indices = []
        for i, sample_id in enumerate(self.clinical_bundle['sample_ids']):
            # Estraiamo il Patient ID (es. TCGA-XX-XXXX) dal Sample ID
            patient_id = sample_id[:12]
            visual_file = self.visual_dir / f"{patient_id}.pt"
            
            if visual_file.exists():
                self.valid_indices.append(i)
            else:
                # Opzionale: log dei pazienti mancanti nel visivo
                # print(f"[DEBUG] Visivo mancante per: {patient_id}")
                pass

        print(f"Dataset Multimodale pronto: {len(self.valid_indices)} campioni accoppiati.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Recuperiamo l'indice originale dello split clinico
        real_idx = self.valid_indices[idx]
        
        # 1. Dati Clinici
        clin_x = self.clinical_bundle['features'][real_idx] # [D_clin_in]
        y = self.clinical_bundle['labels'][real_idx]       # [1]
        
        # 2. Dati Visivi
        patient_id = self.clinical_bundle['sample_ids'][real_idx][:12]
        vis_path = self.visual_dir / f"{patient_id}.pt"
        
        # Carichiamo il file già aggregato [N, 768]
        vis_x = torch.load(vis_path)

        return clin_x, vis_x, y