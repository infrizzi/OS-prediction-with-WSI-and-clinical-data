import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_XLSX = PROJECT_ROOT / "data" / "processed" / "clinical_processed.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    # 1. Caricamento e pulizia
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX)

    # 2. Separazione delle componenti
    # Estraiamo gli ID come lista di stringhe (usiamo Sample ID come chiave primaria)
    ids = df['Sample ID'].values 
    y = df['Overall Survival (Months)'].values.astype('float32')
    X = df.drop(columns=['Patient ID', 'Sample ID', 'Overall Survival (Months)']).values.astype('float32')

    # 3. Primo Split: 70% Train, 30% Temp (Val + Test)
    # Passiamo X, y e ids insieme per mantenere la corrispondenza atomica
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X, y, ids, test_size=0.30, random_state=42
    )

    # 4. Secondo Split: Dividiamo il Temp a metà (15% Val, 15% Test)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp, test_size=0.50, random_state=42
    )

    # 5. Funzione di utilità per creare i "pacchetti" (bundle)
    def create_bundle(X, y, ids):
        return {
            'features': torch.tensor(X),
            'labels': torch.tensor(y).view(-1, 1),
            'sample_ids': list(ids) # Salviamo come lista di stringhe per lookup veloce
        }

    # 6. Salvataggio su disco (3 file totali invece di 11.000)
    torch.save(create_bundle(X_train, y_train, ids_train), OUTPUT_DIR / 'train_data.pt')
    torch.save(create_bundle(X_val, y_val, ids_val), OUTPUT_DIR / 'val_data.pt')
    torch.save(create_bundle(X_test, y_test, ids_test), OUTPUT_DIR / 'test_data.pt')

    print(f"Salvataggio completato in {OUTPUT_DIR}!")
    print(f"Train: {len(ids_train)} | Val: {len(ids_val)} | Test: {len(ids_test)}")

if __name__ == "__main__":
    main()