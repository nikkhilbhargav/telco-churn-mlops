import os
import joblib
import pandas as pd

ARTIFACTS_DIR = "artifacts"

model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))


def predict(data: dict):
    df = pd.DataFrame([data])

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    X = preprocessor.transform(df)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": round(float(probability), 4)
    }