from pathlib import Path
import pandas as pd

# =============================
# 1. CLINICAL DATA CLEANING
# =============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_XLSX = PROJECT_ROOT / "data" / "raw" / "Clinical_data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_XLSX = OUTPUT_DIR / "clinical_clean.xlsx"

PATIENT_COL = "Patient ID"

COLUMNS_TO_KEEP = [
    "Patient ID",
    "Sample ID",
    "Diagnosis Age",
    "Neoplasm Disease Stage American Joint Committee on Cancer Code",
    "Aneuploidy Score",
    "TCGA PanCanAtlas Cancer Type Acronym",
    "Fraction Genome Altered",
    "Genetic Ancestry Label",
    "Neoplasm Histologic Grade",
    "Neoadjuvant Therapy Type Administered Prior To Resection Text",
    "MSI MANTIS Score",
    "MSIsensor Score",
    "Overall Survival (Months)",
    "American Joint Committee on Cancer Metastasis Stage Code",
    "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code",
    "American Joint Committee on Cancer Tumor Stage Code",
    "Primary Lymph Node Presentation Assessment",
    "Prior Diagnosis",
    "Race Category",
    "Sex",
    "Subtype",
    "Tumor Break Load",
    "TMB (nonsynonymous)",
    "Tumor Disease Anatomic Site",
    "Tumor Type",
    "Buffa Hypoxia Score",
]

def onlybrca (dataframe):
    return dataframe[dataframe["TCGA PanCanAtlas Cancer Type Acronym"] == "BRCA"]

def main():
    # Check output dir and read excel file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_XLSX)

    # Excel with ONLY our columns
    df_clean = df[COLUMNS_TO_KEEP].copy()
    df_clean = df_clean[[col for col in df_clean.columns if col != 'Overall Survival (Months)'] + ['Overall Survival (Months)']]
    print(f"Rows with duplicates: {df_clean.shape[0]}")

    # Excel without duplicates
    df_clean_one = df_clean.drop_duplicates(subset=[PATIENT_COL], keep="first").copy()
    print(f"Rows without duplicates: {df_clean_one.shape[0]}")

    # Filter only BRCA samples
    """
    df_clean_one = onlybrca(df_clean_one)
    print(f"Rows with only BRCA: {df_clean_one.shape[0]}")
    """

    # Write the new excel file
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_clean_one.to_excel(writer, index=False, sheet_name="clean_data")

if __name__ == "__main__":
    main()