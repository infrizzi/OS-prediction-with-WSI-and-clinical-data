import os
import json
import torch
import pandas as pd
from pathlib import Path

# ==============================
# Set up the paths (you need to update the paths when files are downloaded)
# ==============================
# Define the path where .pt files are stored (here you provide your custom path)
wsi_data_path = r"C:\Users\feder\Desktop\AI in BioInformatics\progetto\Embedding WSI\pt_files"  # Your custom path to the .pt files

# Path to the Excel file containing clinical data (clinical_clean.xlsx in processed folder)
clinical_data_file = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "clinical_clean.xlsx"  # Path to Excel

# ==============================
# Step 1: Load Clinical Data from Excel
# ==============================
# Load the clinical data from the Excel file
clinical_data = pd.read_excel(clinical_data_file)

# Remove rows where 'Overall Survival' (OS) is missing
# We assume 'OS' column contains the survival information
clinical_data = clinical_data.dropna(subset=['Overall Survival (Months)'])

# ==============================
# Step 2: Create Patient-Slide Association (using Patient ID and Slide file names)
# ==============================
# Dictionary to store the patient-slide associations
patient_slide_dict = {}

# Iterate over the clinical data to associate patients with their slides
for _, row in clinical_data.iterrows():
    patient_id = row['Patient ID']
    
    # Assuming the slide files are named with the pattern 'PatientID-SlideID.pt'
    # and are stored in the wsi_data_path directory
    patient_slide_files = [f for f in os.listdir(wsi_data_path) if f.startswith(str(patient_id)) and f.endswith('.pt')]
    
    # Add the patient-slide association to the dictionary
    if patient_slide_files:
        patient_slide_dict[str(patient_id)] = patient_slide_files

# ==============================
# Step 3: Report - Output Information
# ==============================
# Calculate total number of patients and slides
total_patients = len(patient_slide_dict)
total_slides = sum(len(slides) for slides in patient_slide_dict.values())

# Print some useful stats
print("\n===== Patient-Slide Report =====")
print(f"Total number of patients: {total_patients}")
print(f"Total number of slides: {total_slides}")
print(f"Average number of slides per patient: {total_slides / total_patients:.2f}")

# ==============================
# Step 4: Save the Patient-Slide Association to a JSON file
# ==============================
# Path where the JSON file will be saved (inside the 'processed' folder)
json_output_file = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "patient_slide_association.json"  # Save in processed

# Save the dictionary to a JSON file
with open(json_output_file, 'w') as json_file:
    json.dump(patient_slide_dict, json_file, indent=4)

print("\nPatient-Slide association has been saved to:", json_output_file)
