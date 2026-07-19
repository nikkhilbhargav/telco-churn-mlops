# Telco Customer Churn Prediction (MLOps)

## Features

- Data Preprocessing
- XGBoost Model
- MLflow Experiment Tracking
- FastAPI REST API
- Docker Support

---

## Project Structure

```
telco-churn-mlops/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── feature_columns.json
│
├── data/
│   └── raw/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Train

```bash
python src/train.py
```

---

## Run API

```bash
uvicorn app:app --reload
```

---

## API

GET

```
/
```

POST

```
/predict
```

---

## Tech Stack

- Python
- FastAPI
- XGBoost
- MLflow
- Scikit-learn
- Docker