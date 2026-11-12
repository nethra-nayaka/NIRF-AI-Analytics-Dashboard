from pathlib import Path
import pandas as pd
import re

ALIASES = {
    "year": ["year", "ranking year", "academic year"],
    "category": ["category", "ranking category", "nirf category"],
    "institution": ["institution", "institution name", "name", "institute", "college", "university"],
    "state": ["state", "state name", "location state"],
    "city": ["city", "city name"],
    "rank": ["rank", "rank_num", "ranking", "rank position", "position"],
    "score": ["score", "overall score", "total score", "nirf score"],
    "institution_id": ["institution id", "id", "nirf id", "inst id"],
    "zone": ["zone", "region"],
    "tlr": ["tlr", "teaching learning resources"],
    "rpc": ["rpc", "research professional practice"],
    "go": ["go", "graduation outcomes"],
    "oi": ["oi", "outreach inclusivity"],
    "perception": ["perception", "pr", "perception score"],
}

def normalise_name(name):
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()

def find_col(columns, aliases):
    norm = {normalise_name(c): c for c in columns}
    for alias in aliases:
        if normalise_name(alias) in norm:
            return norm[normalise_name(alias)]
    return None

def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    return pd.read_csv(path, low_memory=False)

def prepare_data(raw):
    df = raw.copy()
    # Map source-specific column names to consistent internal names.
    for target, aliases in ALIASES.items():
        if target not in df.columns:
            col = find_col(df.columns, aliases)
            if col is not None:
                df = df.rename(columns={col: target})
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for c in ["rank","score","tlr","rpc","go","oi","perception"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["category","institution","state","city","zone"]:
        if c in df:
            df[c] = df[c].astype("string").str.strip()
    return df

def profile_data(df):
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(map(str, df.columns)),
        "years": sorted(df["year"].dropna().astype(int).unique().tolist()) if "year" in df else [],
        "categories": sorted(df["category"].dropna().astype(str).unique().tolist()) if "category" in df else [],
        "institutions": int(df["institution"].nunique()) if "institution" in df else None,
    }

def quality_report(df):
    out = pd.DataFrame({
        "column": df.columns.astype(str),
        "data_type": [str(x) for x in df.dtypes],
        "missing_count": [int(x) for x in df.isna().sum()],
        "missing_percent": [round(float(x), 2) for x in (df.isna().mean()*100)],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return out.sort_values(["missing_count","column"], ascending=[False,True])
