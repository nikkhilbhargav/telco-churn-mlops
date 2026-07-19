"""
INFERENCE PIPELINE
"""

import os
import json
import glob
import pandas as pd
import mlflow

# ============================================================
# MODEL LOADING
# ============================================================

MODEL_DIR = "/app/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded from {MODEL_DIR}")

except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")

    local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")

    if not local_model_paths:
        raise Exception("No MLflow model found.")

    latest_model = max(local_model_paths, key=os.path.getmtime)

    model = mlflow.pyfunc.load_model(latest_model)
    MODEL_DIR = latest_model

    print(f"✅ Fallback: Loaded model from {MODEL_DIR}")


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLS = None

# Option 1 (preferred): artifacts/feature_columns.json
json_file = os.path.join("artifacts", "feature_columns.json")

if os.path.exists(json_file):

    with open(json_file, "r") as f:
        FEATURE_COLS = json.load(f)

    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from JSON")

else:

    # Option 2: MLflow artifact
    txt_file = os.path.join(
        os.path.dirname(MODEL_DIR),
        "feature_columns.txt"
    )

    if os.path.exists(txt_file):

        with open(txt_file) as f:
            FEATURE_COLS = [
                x.strip()
                for x in f.readlines()
                if x.strip()
            ]

        print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from MLflow")

    else:
        raise Exception(
            f"feature_columns.json or feature_columns.txt not found.\n"
            f"Tried:\n{json_file}\n{txt_file}"
        )


# ============================================================
# CONSTANTS
# ============================================================

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _serve_transform(df):

    df = df.copy()

    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:

        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    for c, mapping in BINARY_MAP.items():

        if c in df.columns:

            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if obj_cols:
        df = pd.get_dummies(
            df,
            columns=obj_cols,
            drop_first=True
        )

    bool_cols = df.select_dtypes(include=["bool"]).columns

    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(
        columns=FEATURE_COLS,
        fill_value=0
    )

    return df


# ============================================================
# PREDICTION
# ============================================================

def predict(input_dict):

    df = pd.DataFrame([input_dict])

    df = _serve_transform(df)

    pred = model.predict(df)

    if hasattr(pred, "tolist"):
        pred = pred.tolist()

    if isinstance(pred, list):
        pred = pred[0]

    return (
        "Likely to churn"
        if int(pred) == 1
        else "Not likely to churn"
    )