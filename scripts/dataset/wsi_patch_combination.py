import json
import torch
from pathlib import Path
from tqdm import tqdm

# ==============================
# Set up paths
# ==============================
# .pt paths directory
WSI_PATH = Path(r"/work/h2020deciderficarra_shared/TCGA/BRCA/features_CONCH/pt_files")

# Output directory
OUTPUT_DIR = Path(r"/homes/lpaladino/first_combination")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# JSON file path
JSON_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "patient_slide_association.json"

def main():

    # ==============================
    # JSON Loading
    # ==============================
    with open(JSON_FILE, 'r') as f:
        patient_dict = json.load(f)

    print(f"Starting first combination for each {len(patient_dict)} patient's slides...")

    # ==============================
    # First aggregation loop
    # ==============================
    for patient_id, data in tqdm(patient_dict.items(), desc="Combining slides"):
        slide_files = data['slides']
        
        #print(f"Processing: {patient_id}")

        patient_tensors = []
        
        for slide_name in slide_files:
            slide_path = WSI_PATH / slide_name
            
            if slide_path.exists():
                tensor = torch.load(slide_path, map_location='cpu')
                # print(tensor.shape)
                patient_tensors.append(tensor)

            else:
                print(f"\n[WARNING] File not found: {slide_name}")

        
        if patient_tensors:
            # Step 1: Combining tensors along dimension 0
            combined_tensor = torch.cat(patient_tensors, dim=0)
            
            # Step 2: Saving the combined tensor
            output_file = OUTPUT_DIR / f"{patient_id}.pt"
            
            torch.save(combined_tensor, output_file)
        else:
            print(f"\n[ERROR] No tensor found for patient {patient_id}")
        
    print(f"\nCombination complete, output saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 