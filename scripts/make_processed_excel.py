from pathlib import Path
import pandas as pd
import numpy as np

# ==========================
# CONFIG
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_XLSX = PROJECT_ROOT / "data" / "processed" / "clinical_clean.xlsx"
OUTPUT_XLSX = PROJECT_ROOT / "data" / "processed" / "clinical_processed.xlsx"

SHEET_IN = "clean_data"

ID_COLS = ["Patient ID", "Sample ID"]
LABEL_COL = "Overall Survival (Months)"
CANCER_TYPE_COL = "TCGA PanCanAtlas Cancer Type Acronym"

# Numeric columns whose median must be computed per cancer type (group-wise)
GROUPWISE_MEDIAN_NUMERIC_COLS = [
    "MSI MANTIS Score",
    "MSIsensor Score",
    "TMB (nonsynonymous)"
]

# Columns that are often binary in TCGA-style clinical tables (we will auto-detect anyway)
# If a column looks binary, we treat it as binary.
POTENTIAL_BINARY_COLS = [
    "Prior Diagnosis",
]

# Choose scaling: StandardScaler (recommended)
USE_STANDARD_SCALER = True  # if False, uses MinMax 0-1 scaling


# ==========================
# HELPERS
# ==========================
def normalize_str(x: str) -> str:
    return " ".join(str(x).strip().split()).lower()

def is_binary_series(s: pd.Series) -> bool:
    """Heuristic: checks whether a column is binary-like (0/1, yes/no, true/false)."""
    if s.dropna().empty:
        return False
    vals = s.dropna().unique()

    # If numeric and only {0,1}
    if pd.api.types.is_numeric_dtype(s):
        unique = set(pd.Series(vals).astype(float).unique())
        return unique.issubset({0.0, 1.0}) and len(unique) <= 2

    # If string-like: map normalized strings
    mapped = set()
    for v in vals:
        vv = normalize_str(v)
        mapped.add(vv)

    allowed = {"0", "1", "yes", "no", "true", "false", "y", "n"}
    return mapped.issubset(allowed) and len(mapped) <= 2

def to_binary_01(s: pd.Series) -> pd.Series:
    """Convert binary-like values to 0/1 (keeps NaN)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float")
    # strings
    def conv(v):
        if pd.isna(v):
            return np.nan
        vv = normalize_str(v)
        if vv in {"1", "yes", "true", "y"}:
            return 1.0
        if vv in {"0", "no", "false", "n"}:
            return 0.0
        return np.nan  # if weird values, treat as missing
    return s.map(conv).astype("float")

def safe_standardize(col: pd.Series) -> pd.Series:
    """StandardScaler-like (z-score). If std == 0, returns 0."""
    mu = col.mean()
    sigma = col.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(col)), index=col.index)
    return (col - mu) / sigma

def safe_minmax(col: pd.Series) -> pd.Series:
    """MinMax scaling 0..1. If constant, returns 0."""
    mn = col.min()
    mx = col.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(col)), index=col.index)
    return (col - mn) / (mx - mn)

def drop_missing_os(df, os_col):
    """
    Remove rows with missing Overall Survival.
    """
    n_before = len(df)
    df_clean = df.dropna(subset=[os_col]).copy()
    n_after = len(df_clean)

    print("\n" + "-" * 80)
    print("OVERALL SURVIVAL CLEANING")
    print("-" * 80)
    print(f"Rows before OS cleaning: {n_before}")
    print(f"Rows after  OS cleaning: {n_after}")
    print(f"Removed rows (missing OS): {n_before - n_after}")
    print("-" * 80)

    return df_clean


# ==========================
# MAIN
# ==========================
def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_XLSX}")

    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_IN)
    df = drop_missing_os(df, LABEL_COL)

    # Basic checks
    for c in ID_COLS + [LABEL_COL, CANCER_TYPE_COL]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # Separate parts
    df_id = df[ID_COLS].copy()
    y = df[LABEL_COL].copy()

    # Features (everything except ID + label)
    feature_cols = [c for c in df.columns if c not in set(ID_COLS + [LABEL_COL])]
    X = df[feature_cols].copy()

    # Detect column types
    binary_cols = []
    numeric_cols = []
    categorical_cols = []

    for c in feature_cols:
        s = X[c]
        if is_binary_series(s):
            binary_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    # Ensure groupwise cols are treated numeric (even if loaded as object)
    # Try coercion to numeric
    for c in GROUPWISE_MEDIAN_NUMERIC_COLS:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if c not in numeric_cols and c not in binary_cols:
                # if coercion makes it numeric, add to numeric
                if pd.api.types.is_numeric_dtype(X[c]):
                    numeric_cols.append(c)
                    if c in categorical_cols:
                        categorical_cols.remove(c)

    # --- Process BINARY columns: missing -> 0 + missing flag
    X_bin = pd.DataFrame(index=X.index)
    bin_missing_flags = pd.DataFrame(index=X.index)

    for c in binary_cols:
        s = to_binary_01(X[c])
        miss = s.isna().astype(int).rename(f"{c}__missing")
        s_filled = s.fillna(0.0).rename(c)
        X_bin[c] = s_filled
        bin_missing_flags[miss.name] = miss

    # --- Process NUMERIC columns: median imputation (+ groupwise for selected) + missing flag
    X_num = pd.DataFrame(index=X.index)
    num_missing_flags = pd.DataFrame(index=X.index)

    # Global medians
    global_medians = {}
    for c in numeric_cols:
        global_medians[c] = pd.to_numeric(X[c], errors="coerce").median()

    # Groupwise medians for selected cols
    group_medians = {}
    for c in GROUPWISE_MEDIAN_NUMERIC_COLS:
        if c in X.columns:
            gm = (
                pd.concat([df[CANCER_TYPE_COL], pd.to_numeric(X[c], errors="coerce")], axis=1)
                .groupby(CANCER_TYPE_COL)[c]
                .median()
            )
            group_medians[c] = gm

    for c in numeric_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        miss = s.isna().astype(int).rename(f"{c}__missing")

        if c in group_medians:
            # fill by cancer type median, fallback to global median
            s_filled = s.copy()
            # map group median to each row
            mapped = df[CANCER_TYPE_COL].map(group_medians[c])
            s_filled = s_filled.fillna(mapped)
            s_filled = s_filled.fillna(global_medians[c])
        else:
            s_filled = s.fillna(global_medians[c])

        X_num[c] = s_filled
        num_missing_flags[miss.name] = miss

    # --- Process CATEGORICAL columns: one-hot with unknown category
    # Fill NaN with "unknown" and ensure "unknown" category exists
    X_cat = pd.DataFrame(index=X.index)

    for c in categorical_cols:
        s = X[c].astype("object").where(~X[c].isna(), "unknown")
        # force unknown even if none missing by adding category manually via Categorical
        # (not strictly necessary, but makes it explicit)
        cats = sorted(set(s.unique().tolist() + ["unknown"]))
        s = pd.Categorical(s, categories=cats)
        dummies = pd.get_dummies(s, prefix=c, prefix_sep="=")
        X_cat = pd.concat([X_cat, dummies], axis=1)

    # --- Combine all feature blocks
    X_processed = pd.concat(
        [X_num, num_missing_flags, X_bin, bin_missing_flags, X_cat],
        axis=1
    )

    # --- Scale numeric columns (ONLY true numeric values, not missing flags, not one-hot)
    # We'll scale X_num only, then replace in X_processed
    if USE_STANDARD_SCALER:
        X_num_scaled = X_num.apply(safe_standardize, axis=0)
    else:
        X_num_scaled = X_num.apply(safe_minmax, axis=0)

    for c in X_num.columns:
        X_processed[c] = X_num_scaled[c]

    # --- Output dataframe: ID + processed features + label
    out = pd.concat([df_id, X_processed, y.rename(LABEL_COL)], axis=1)

    # --- Build a feature map sheet (useful for checking)
    feature_map_rows = []
    for c in numeric_cols:
        feature_map_rows.append((c, "numeric", "kept", c))
        feature_map_rows.append((c, "missing_flag", "added", f"{c}__missing"))
    for c in binary_cols:
        feature_map_rows.append((c, "binary", "kept", c))
        feature_map_rows.append((c, "missing_flag", "added", f"{c}__missing"))
    for c in categorical_cols:
        # list the one-hot columns created
        onehot_cols = [cc for cc in X_cat.columns if cc.startswith(f"{c}=")]
        for oh in onehot_cols:
            feature_map_rows.append((c, "categorical_onehot", "created", oh))

    feature_map = pd.DataFrame(feature_map_rows, columns=["original_column", "type", "action", "output_column"])

    # Save
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="processed_data")
        feature_map.to_excel(writer, index=False, sheet_name="feature_map")

    # Print small report
    print("\n" + "=" * 80)
    print("PROCESSED EXCEL CREATED")
    print("=" * 80)
    print(f"Input file:   {INPUT_XLSX}")
    print(f"Output file:  {OUTPUT_XLSX}")
    print(f"Rows:         {len(out)}")
    print(f"Output cols:  {out.shape[1]}")
    print(f"Numeric cols: {len(numeric_cols)}  (scaled: {'StandardScaler' if USE_STANDARD_SCALER else 'MinMax'})")
    print(f"Binary cols:  {len(binary_cols)}")
    print(f"Cat cols:     {len(categorical_cols)}  (one-hot expanded to {X_cat.shape[1]} cols)")
    print("=" * 80)

    # Optional: show which columns were detected as binary
    if binary_cols:
        print("\nDetected binary columns:")
        for c in binary_cols:
            print(f"  - {c}")

if __name__ == "__main__":
    main()