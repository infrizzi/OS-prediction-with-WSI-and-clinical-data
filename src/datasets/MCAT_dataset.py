import torch
from torch.utils.data import Dataset
from pathlib import Path

class MCATDataset(Dataset):
    def __init__(self, clinical_split_path, visual_dir):
        """
        Args:
            clinical_split_path (str/Path): path clinical .pt file
            visual_dir (str/Path): directory visual split
        """
        # 1. Load clinical bundle
        # Structure: {'features': tensor, 'labels': tensor, 'sample_ids': list}
        self.clinical_bundle = torch.load(clinical_split_path)
        self.visual_dir = Path(visual_dir)
        
        # 2. Take valid indices where both clinical and visual data exist
        self.valid_indices = []
        for i, sample_id in enumerate(self.clinical_bundle['sample_ids']):
            # Extracting patient ID from sample ID
            patient_id = sample_id[:12]
            visual_file = self.visual_dir / f"{patient_id}.pt"
            
            if visual_file.exists():
                self.valid_indices.append(i)

        print(f"Multimodal dataset ready: {len(self.valid_indices)} samples.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Original index in the clinical bundle
        real_idx = self.valid_indices[idx]
        
        # 1. Clinical data
        clin_x = self.clinical_bundle['features'][real_idx] # [D_clin_in]
        y = self.clinical_bundle['labels'][real_idx]       # [1]
        
        # 2. Visual data
        patient_id = self.clinical_bundle['sample_ids'][real_idx][:12]
        vis_path = self.visual_dir / f"{patient_id}.pt"
        
        # Load visual feature
        vis_x = torch.load(vis_path) # [N_patches, D_vis_in]

        return clin_x, vis_x, y