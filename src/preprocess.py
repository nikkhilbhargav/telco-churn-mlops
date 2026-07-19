import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Artifact directory
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the Telco Customer Churn dataset.

    Returns:
        X_processed
        y
        fitted_preprocessor
    """

    df = df.copy()

    # Remove Customer ID
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # Convert target to numeric
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Detect column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Numerical pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    # Save fitted preprocessor
    joblib.dump(
        preprocessor,
        os.path.join(
            ARTIFACTS_DIR,
            "preprocessor.pkl"
        )
    )

    # Save transformed feature names
    feature_columns = preprocessor.get_feature_names_out().tolist()

    with open(
        os.path.join(
            ARTIFACTS_DIR,
            "feature_columns.json"
        ),
        "w"
    ) as f:
        json.dump(feature_columns, f, indent=4)

    print("✅ Preprocessor saved successfully.")
    print("✅ Feature columns saved successfully.")

    return X_processed, y, preprocessor