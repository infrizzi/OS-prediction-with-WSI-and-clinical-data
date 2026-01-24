from pathlib import Path
import pandas as pd

# ==========================
# CONFIG
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_XLSX = PROJECT_ROOT / "data" / "raw" / "Clinical_data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_XLSX = OUTPUT_DIR / "clinical_clean.xlsx"

PATIENT_COL = "Patient ID"
SAMPLE_COL = "Sample ID"

# Columns to keep (everything else will be dropped)
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

# Column grouping for documentation (sheet "column_groups")
ID_COLS = ["Patient ID", "Sample ID"]
LABEL_COLS = ["Overall Survival (Months)"]
FEATURE_COLS = [c for c in COLUMNS_TO_KEEP if c not in ID_COLS + LABEL_COLS]


# ==========================
# HELPERS
# ==========================
def normalize(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


def resolve_columns(df_cols, wanted_cols):
    """
    Try exact match first; if not found, try normalized match.
    Returns:
      resolved_map: dict(wanted -> actual)
      missing: list of wanted cols not found
    """
    df_cols_list = list(df_cols)
    df_norm_map = {normalize(c): c for c in df_cols_list}

    resolved_map = {}
    missing = []

    for w in wanted_cols:
        if w in df_cols_list:
            resolved_map[w] = w
        else:
            wn = normalize(w)
            if wn in df_norm_map:
                resolved_map[w] = df_norm_map[wn]
            else:
                missing.append(w)

    return resolved_map, missing


# ==========================
# MAIN
# ==========================
def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_XLSX}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(INPUT_XLSX)

    # Resolve column names robustly (handles extra spaces/case differences)
    resolved_map, missing = resolve_columns(df.columns, COLUMNS_TO_KEEP)

    if missing:
        print("\n[WARNING] Some requested columns were NOT found in the input file:")
        for m in missing:
            print(f"  - {m}")
        print("\nAvailable columns in the file:")
        for c in df.columns:
            print(f"  - {c}")
        raise KeyError("Missing required columns. Fix the column names or the input file headers.")

    # Rename columns to the canonical names (the ones in COLUMNS_TO_KEEP)
    # so downstream scripts always see consistent names.
    rename_dict = {actual: wanted for wanted, actual in resolved_map.items() if actual != wanted}
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # 1) Keep only 1 sample per patient (deterministic):
    # sort by Sample ID and keep first row per patient
    df_sorted = df.sort_values(by=[PATIENT_COL, SAMPLE_COL], kind="mergesort")
    df_one_sample = df_sorted.drop_duplicates(subset=[PATIENT_COL], keep="first").copy()

    # 2) Drop everything except the selected columns
    df_clean = df_one_sample[COLUMNS_TO_KEEP].copy()

    # 3) Reorder: ID -> FEATURES -> LABEL
    ordered_cols = ID_COLS + FEATURE_COLS + LABEL_COLS
    df_clean = df_clean[ordered_cols]

    # 4) Save excel (two sheets: data + column_groups)
    groups_df = pd.DataFrame({
        "column": ordered_cols,
        "group": (
            ["ID"] * len(ID_COLS)
            + ["FEATURE"] * len(FEATURE_COLS)
            + ["LABEL"] * len(LABEL_COLS)
        ),
    })

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_clean.to_excel(writer, index=False, sheet_name="clean_data")
        groups_df.to_excel(writer, index=False, sheet_name="column_groups")

    # Small report
    n_before = df[PATIENT_COL].nunique()
    n_after = df_clean[PATIENT_COL].nunique()
    print("\n" + "=" * 80)
    print("CLEAN EXCEL CREATED")
    print("=" * 80)
    print(f"Input rows:  {len(df)}")
    print(f"Output rows: {len(df_clean)}")
    print(f"Unique patients before: {n_before}")
    print(f"Unique patients after:  {n_after}")
    print(f"Saved to: {OUTPUT_XLSX}")
    print("=" * 80)


if __name__ == "__main__":
    main()