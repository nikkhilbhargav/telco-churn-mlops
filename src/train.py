import os
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocess import preprocess_data


DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("Telco-Churn-MLOps")

    with mlflow.start_run():

        print("Training Model...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        cm = confusion_matrix(y_test, predictions)

        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Parameters
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", 5)

        # MLflow Model
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model"
        )

        # Local Model
        joblib.dump(
            model,
            os.path.join(
                ARTIFACTS_DIR,
                "model.pkl"
            )
        )

        print("\nModel Saved Successfully\n")

        print("=" * 60)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        print("=" * 60)

        print("\nConfusion Matrix\n")

        print(cm)

        print("\nArtifacts Generated\n")

        print("✔ model.pkl")
        print("✔ preprocessor.pkl")
        print("✔ feature_columns.json")

        print("=" * 60)


if __name__ == "__main__":
    train()