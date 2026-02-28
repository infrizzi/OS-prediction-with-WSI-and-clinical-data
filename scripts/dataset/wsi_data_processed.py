import os
import json
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

# ==============================
# Set up paths
# ==============================
WSI_PATH = r"/work/h2020deciderficarra_shared/TCGA/BRCA/features_CONCH/pt_files" 

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLINICAL_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_clean.xlsx"
JSON_FILE = PROJECT_ROOT / "data" / "processed" / "patient_slide_association.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    # ==============================
    # Step 1: Load Clinical Data
    # ==============================
    clinical_data = pd.read_excel(CLINICAL_PATH)

    # Remove NA OS entries
    clinical_data = clinical_data.dropna(subset=['Overall Survival (Months)'])

    # ==============================
    # Step 2: Create Patient-Slide-OS Association
    # ==============================
    patient_slide_dict = {}

    # Scaling OS values
    scaler = StandardScaler()
    clinical_data['Overall Survival (Months)'] = scaler.fit_transform(clinical_data[['Overall Survival (Months)']])

    for _, row in clinical_data.iterrows():
        patient_id = str(row['Patient ID'])
        os_months = row['Overall Survival (Months)']
        
        # Looking for slide files corresponding to the patient ID
        patient_slide_files = [f for f in os.listdir(WSI_PATH) if f.startswith(patient_id) and f.endswith('.pt')]
        
        if patient_slide_files:
            patient_slide_dict[patient_id] = {
                "slides": patient_slide_files,
                "os_months": float(os_months)
            }

    # ==============================
    # Step 3: Report - Output Information
    # ==============================
    total_patients = len(patient_slide_dict)
    total_slides = sum(len(info["slides"]) for info in patient_slide_dict.values())

    print("\n===== Patient-Slide Report =====")
    print(f"Total number of patients with slides and OS: {total_patients}")
    print(f"Total number of slides: {total_slides}")
    if total_patients > 0:
        print(f"Average number of slides per patient: {total_slides / total_patients:.2f}")

    # ==============================
    # Step 4: Save to JSON
    # ==============================
    joblib.dump(scaler, OUTPUT_DIR / "target_scaler_wsi.pkl") # Save scaler for labels

    with open(JSON_FILE, 'w') as json_file:
        json.dump(patient_slide_dict, json_file, indent=4)

    print("\nAssociation saved in:", JSON_FILE)

if __name__ == "__main__":
    main()