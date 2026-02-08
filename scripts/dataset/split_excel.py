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

    # Identifichiamo i nomi delle colonne delle feature per trovare l'indice di BRCA
    feature_names = df.drop(columns=['Patient ID', 'Sample ID', 'Overall Survival (Months)']).columns
    
    # Separating components
    ids = df['Sample ID'].values 
    y = df['Overall Survival (Months)'].values.astype('float32')
    X = df.drop(columns=['Patient ID', 'Sample ID', 'Overall Survival (Months)']).values.astype('float32')

# 1. Creiamo la colonna di stratificazione originale
    strat_col = df['TCGA PanCanAtlas Cancer Type Acronym_BRCA'].values

    # First split: 70% Train, 30% Temp (Val + Test)
    # Aggiungiamo 'strat_col' agli array da splittare
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp, strat_train, strat_temp = train_test_split(
        X, y, ids, strat_col, 
        test_size=0.30, 
        random_state=42, 
        stratify=strat_col # Qui usiamo quella originale
    )

    # Second split: cut Temp in half (15% Val, 15% Test)
    # Qui dobbiamo usare 'strat_temp' che è stata splittata correttamente (lunga 3242)
    X_val, X_test, y_val, y_test, ids_val, ids_test, strat_val, strat_test = train_test_split(
        X_temp, y_temp, ids_temp, strat_temp,
        test_size=0.50, 
        random_state=42, 
        stratify=strat_temp # Fondamentale: deve avere la stessa lunghezza di X_temp
    )

    # =============================================================
    # CONTEGGIO CAMPIONI BRCA
    # =============================================================
    target_col = 'TCGA PanCanAtlas Cancer Type Acronym_BRCA'
    
    if target_col in feature_names:
        # Troviamo la posizione della colonna nella matrice X
        brca_idx = list(feature_names).index(target_col)
        
        # Contiamo dove il valore è 1 nelle matrici X di ogni split
        n_brca_train = (X_train[:, brca_idx] == 1).sum()
        n_brca_val   = (X_val[:, brca_idx] == 1).sum()
        n_brca_test  = (X_test[:, brca_idx] == 1).sum()
        
        print(f"\nConteggio BRCA:")
        print(f"Train: {int(n_brca_train)} | Val: {int(n_brca_val)} | Test: {int(n_brca_test)}")
    else:
        print(f"\n[WARNING] Colonna '{target_col}' non trovata nel dataset.")
    # =============================================================

    
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

    print(f"\nSalvataggio completato in {OUTPUT_DIR}!")
    print(f"Dataset Totale - Train: {len(ids_train)} | Val: {len(ids_val)} | Test: {len(ids_test)}")
    
if __name__ == "__main__":
    main()