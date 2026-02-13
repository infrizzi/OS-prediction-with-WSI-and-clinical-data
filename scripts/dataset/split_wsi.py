import torch
import shutil
from pathlib import Path
from tqdm import tqdm

# =============================================================================================================
# N.B. This scipts MIRRORS the clinical splits into visual splits by copying the relevant visual tensors.
#      For this reason, you need to follow these passages:
#      1) Run 'split_excel.py' to create clinical splits (train/val/test) based on clinical data
#      2) Run this script to create visual splits that mirror the clinical ones
#      
#      In this way, you ensure that both clinical and visual data are aligned for training/validation/testing.
# =============================================================================================================

# ==========================================
# 1. SET UP PATHS
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLINICAL_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
VISUAL_AGGREGATED_DIR = Path(r"C:\Users\lucap\Downloads\File_FBI\first_combination")

# New output directory for visual splits
OUTPUT_VISUAL_SPLITS = Path(r"C:\Users\lucap\Downloads\File_FBI\visual_splits")
OUTPUT_VISUAL_SPLITS_MULTIMODAL = Path(r"C:\Users\lucap\Downloads\File_FBI\visual_splits_multimodal")

def create_visual_split(split_name, output_dir):
    print(f"\nProcessing {split_name} split...")
    
    # Load clinical split data to get sample IDs
    clinical_data = torch.load(CLINICAL_SPLITS_DIR / f'{split_name}_data.pt')
    sample_ids = clinical_data['sample_ids']
    
    # Destination directory for the visual split
    dest_dir = output_dir / split_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    missing_count = 0
    for s_id in tqdm(sample_ids):
        # Mapping sample ID to patient ID
        p_id = s_id[:12] if s_id.startswith("TCGA") else s_id
        
        source_file = VISUAL_AGGREGATED_DIR / f"{p_id}.pt"
        target_file = dest_dir / f"{p_id}.pt"
        
        if source_file.exists():
            shutil.copy(source_file, target_file)

def main():
    # main function to create visual splits
    for split in ['train', 'val', 'test']:
        create_visual_split(split, OUTPUT_VISUAL_SPLITS)
    
    print(f"\nMirroring completed! Files are in: {OUTPUT_VISUAL_SPLITS}")

if __name__ == "__main__":
    main()