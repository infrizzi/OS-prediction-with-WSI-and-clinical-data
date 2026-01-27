import pandas as pd
import numpy as np
from pathlib import Path


# ==========================
# CONFIG
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FILE_PATH = PROJECT_ROOT / "data" / "raw" / "Clinical_data.xlsx"
PATIENT_COL = "Patient ID"
SAMPLE_COL = "Sample ID"

# ==========================
# LOAD
# ==========================
df = pd.read_excel(FILE_PATH)

print("\n" + "="*80)
print("CLINICAL DATASET OVERVIEW")
print("="*80)

# ==========================
# BASIC SHAPE
# ==========================
print(f"\n[1] Dataset shape")
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# ==========================
# PATIENT vs SAMPLE ANALYSIS
# ==========================
print("\n" + "="*80)
print("[2] PATIENT / SAMPLE CARDINALITY")
print("="*80)

n_patients = df[PATIENT_COL].nunique()
n_samples = df[SAMPLE_COL].nunique()

print(f"Unique patients: {n_patients}")
print(f"Unique samples:  {n_samples}")

samples_per_patient = (
    df.groupby(PATIENT_COL)[SAMPLE_COL]
      .nunique()
      .sort_values(ascending=False)
)

print("\nSamples per patient (summary):")
print(samples_per_patient.describe())

print("\nPatients with >1 sample:")
print((samples_per_patient > 1).sum())

print("\nTop patients by number of samples:")
print(samples_per_patient.head(10))

# ==========================
# MISSING VALUES ANALYSIS
# ==========================
print("\n" + "="*80)
print("[3] MISSING VALUES PER COLUMN")
print("="*80)

missing_counts = df.isna().sum()
missing_pct = (missing_counts / len(df)) * 100

missing_df = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct
}).sort_values("missing_pct", ascending=False)

print(missing_df.head(15))

# ==========================
# COLUMN TYPE ANALYSIS
# ==========================
print("\n" + "="*80)
print("[4] COLUMN TYPE ANALYSIS")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}):")
for c in numeric_cols:
    print(f"  - {c}")

print(f"\nCategorical / string columns ({len(categorical_cols)}):")
for c in categorical_cols:
    print(f"  - {c}")

# ==========================
# LOW-VARIANCE / CONSTANT COLUMNS
# ==========================
print("\n" + "="*80)
print("[5] LOW VARIANCE / CONSTANT COLUMNS")
print("="*80)

low_variance_cols = []

for c in df.columns:
    if df[c].nunique(dropna=True) <= 1:
        low_variance_cols.append(c)

if low_variance_cols:
    print("Columns with <=1 unique value:")
    for c in low_variance_cols:
        print(f"  - {c}")
else:
    print("No constant columns found.")

# ==========================
# CATEGORICAL CARDINALITY
# ==========================
print("\n" + "="*80)
print("[6] CATEGORICAL CARDINALITY (top levels)")
print("="*80)

for c in categorical_cols:
    n_unique = df[c].nunique(dropna=True)
    print(f"\nColumn: {c}")
    print(f"Unique values: {n_unique}")
    print(df[c].value_counts(dropna=False).head(5))

# ==========================
# NUMERIC STATS
# ==========================
print("\n" + "="*80)
print("[7] NUMERIC FEATURE STATISTICS")
print("="*80)

print(df[numeric_cols].describe().T)

# ==========================
# POTENTIAL LEAKAGE CHECK
# ==========================
print("\n" + "="*80)
print("[8] POTENTIAL LEAKAGE COLUMNS (NAME-BASED HEURISTIC)")
print("="*80)

leakage_keywords = [
    "alive", "death", "follow", "survival",
    "last", "disease free", "dfs", "os"
]

for c in df.columns:
    if any(k in c.lower() for k in leakage_keywords):
        print(f"  - {c}")

print("\nAnalysis completed.")
print("="*80)