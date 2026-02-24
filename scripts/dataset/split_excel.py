import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

# =============================
# 3. SPLIT CLINICAL DATA
# =============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_XLSX = PROJECT_ROOT / "data" / "processed" / "clinical_processed_new.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

def main():
    # Check output dir and read excel file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX)

    # Keeping only features
    feature_names = df.drop(columns=['Patient ID', 'Sample ID', 'Overall Survival (Months)']).columns
    
    # Separating components
    ids = df['Sample ID'].values 
    y = df['Overall Survival (Months)'].values.astype('float32')
    X = df.drop(columns=['Patient ID', 'Sample ID', 'Overall Survival (Months)']).values.astype('float32')

    # Creating training, validation and test splits with stratification on the BRCA column
    strat_col = df['TCGA PanCanAtlas Cancer Type Acronym_BRCA'].values

    # First split: 70% Train, 30% Temp (Val + Test) + stratify by BRCA column
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp, strat_train, strat_temp = train_test_split(
        X, y, ids, strat_col, 
        test_size=0.30, 
        random_state=42, 
        stratify=strat_col # Original stratification
    )

    # Second split: cut Temp in half (15% Val, 15% Test) + splitted stratification
    X_val, X_test, y_val, y_test, ids_val, ids_test, strat_val, strat_test = train_test_split(
        X_temp, y_temp, ids_temp, strat_temp,
        test_size=0.50, 
        random_state=42, 
        stratify=strat_temp # Splitted stratification
    )

    # =============================================================
    # BRCA COUNT CHECK
    # =============================================================
    target_col = 'TCGA PanCanAtlas Cancer Type Acronym_BRCA'
    
    if target_col in feature_names:
        # Find the index of the BRCA column in the feature names
        brca_idx = list(feature_names).index(target_col)
        
        # Count the number of samples with BRCA = 1 in each split
        n_brca_train = (X_train[:, brca_idx] == 1).sum()
        n_brca_val   = (X_val[:, brca_idx] == 1).sum()
        n_brca_test  = (X_test[:, brca_idx] == 1).sum()
        
        print(f"\nBRCA Counting:")
        print(f"Train: {int(n_brca_train)} | Val: {int(n_brca_val)} | Test: {int(n_brca_test)}")
    else:
        print(f"\n[WARNING] Column '{target_col}' not found in features.")
    # =============================================================

    
    # Bundle creation function (ids kept for future WSI aggregation)
    def create_bundle(X, y, ids):
        return {
            'features': torch.tensor(X),
            'labels': torch.tensor(y).view(-1, 1),
            'sample_ids': list(ids)
        }
    
    # Save bundles (train, val, test)
    torch.save(create_bundle(X_train, y_train, ids_train), OUTPUT_DIR / 'train_data_new.pt')
    torch.save(create_bundle(X_val, y_val, ids_val), OUTPUT_DIR / 'val_data_new.pt')
    torch.save(create_bundle(X_test, y_test, ids_test), OUTPUT_DIR / 'test_data_new.pt')

    print(f"\nSaved bundles to {OUTPUT_DIR}!")
    print(f"Total dataset - Train: {len(ids_train)} | Val: {len(ids_val)} | Test: {len(ids_test)}")
    
if __name__ == "__main__":
    main()