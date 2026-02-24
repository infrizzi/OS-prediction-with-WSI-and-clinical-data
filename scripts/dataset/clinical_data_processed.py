from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =============================
# 2. CLINICAL DATA PROCESSING
# =============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_XLSX = PROJECT_ROOT / "data" / "processed" / "clinical_clean.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_XLSX = PROJECT_ROOT / "data" / "processed" / "clinical_processed_new.xlsx"

ID_COLS = ["Patient ID", "Sample ID"]
LABEL_COL = "Overall Survival (Months)"
CANCER_TYPE_COL = "TCGA PanCanAtlas Cancer Type Acronym"

GROUPWISE_MEDIAN_NUMERIC_COLS = [
    "MSI MANTIS Score",
    "MSIsensor Score",
    "TMB (nonsynonymous)"
]


def select_top_features(df, target_col, n_features=49):
    # Separiamo feature e target
    X = df.drop(columns=ID_COLS + [target_col])
    y = df[target_col]
    
    # Alleniamo un RF veloce per pesare le feature
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Prendiamo i nomi delle top N feature
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_cols = importances.sort_values(ascending=False).head(n_features).index.tolist()
    
    print(f"{n_features} important features selected")
    # Restituiamo il dataframe con solo ID, Top Features e Label
    return df[ID_COLS + top_cols + [target_col]]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX)

    df_id = df[ID_COLS].copy()
    label = df[LABEL_COL].copy()

    features = [c for c in df.columns if c not in set(ID_COLS + [LABEL_COL])]
    X = df[features].copy()

    categorical_cols = [cname for cname in features if X[cname].dtype == "object"]
    numerical_cols = [cname for cname in features if X[cname].dtype in ['int64', 'float64']]
    
    # 1. Missing flag creation for numerical cols
    for col in numerical_cols:
        if X[col].isna().any():
            X[f"{col}_missing"] = X[col].isna().astype(int)

    # 2. Fill NA
    for column in X.columns:
        if column in categorical_cols:
            X[column] = X[column].fillna('unknown')
        elif column in numerical_cols:
            if column not in GROUPWISE_MEDIAN_NUMERIC_COLS:
                X[column] = X[column].fillna(X[column].median())
            else:
                gm = X.groupby(CANCER_TYPE_COL)[column].median()
                X[column] = X[column].fillna(X[CANCER_TYPE_COL].map(gm)).fillna(X[column].median())

    # 3. One-Hot Encoding
    X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols, dtype=int)

    # 4. StandardScaler
    scaler = StandardScaler()
    for column in numerical_cols:
        X[column] = scaler.fit_transform(X[[column]])

    # Rebuild dataframe and drop NA label
    df_out = pd.concat([df_id, X, label], axis=1)
    df_out = df_out.dropna(subset=['Overall Survival (Months)'])

    df_out = select_top_features(df_out, LABEL_COL, n_features=49)
    df_out['TCGA PanCanAtlas Cancer Type Acronym_BRCA'] = X['TCGA PanCanAtlas Cancer Type Acronym_BRCA']

    df_out[LABEL_COL] = scaler.fit_transform(df_out[[LABEL_COL]])
    joblib.dump(scaler, OUTPUT_DIR / "target_scaler.pkl") # Save scaler for labels

    # Export
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="clean_data")
    
    print(f"File saved in: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()