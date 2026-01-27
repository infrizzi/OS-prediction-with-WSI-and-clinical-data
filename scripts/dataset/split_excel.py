import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

# =============================
# 3. SPLIT CLINICAL DATA
# =============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_XLSX = PROJECT_ROOT / "data" / "processed" / "clinical_processed.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

def main():
    # Check output dir and read excel file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX)

    # Separating components
    ids = df['Sample ID'].values 
    y = df['Overall Survival (Months)'].values.astype('float32')
    X = df.drop(columns=['Patient ID', 'Sample ID', 'Overall Survival (Months)']).values.astype('float32')

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X, y, ids, test_size=0.30, random_state=42
    )

    # Second split: cut Temp in half (15% Val, 15% Test)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp, test_size=0.50, random_state=42
    )

    # Bundle creation function (ids kept for future WSI aggregation)
    def create_bundle(X, y, ids):
        return {
            'features': torch.tensor(X),
            'labels': torch.tensor(y).view(-1, 1),
            'sample_ids': list(ids)
        }

    # Save bundles (train, val, test)
    torch.save(create_bundle(X_train, y_train, ids_train), OUTPUT_DIR / 'train_data.pt')
    torch.save(create_bundle(X_val, y_val, ids_val), OUTPUT_DIR / 'val_data.pt')
    torch.save(create_bundle(X_test, y_test, ids_test), OUTPUT_DIR / 'test_data.pt')

    print(f"Salvataggio completato in {OUTPUT_DIR}!")
    print(f"Train: {len(ids_train)} | Val: {len(ids_val)} | Test: {len(ids_test)}")

if __name__ == "__main__":
    main()