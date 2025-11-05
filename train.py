import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import mlflow

load_dotenv()
mlflow.set_experiment("Titanic Baseline Models")

train_df = pd.read_csv("data/train.csv")
X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]

model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
model_path = "models/logreg_model.pkl"
joblib.dump(model, model_path)

train_acc = accuracy_score(y_train, model.predict(X_train))

with open("train_logreg_metrics.json", "w") as f:
    json.dump({"train_accuracy": train_acc}, f, indent=4)

with mlflow.start_run(run_name="Titanic_LogisticRegression"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact("train_logreg_metrics.json")

print(f"✅ Logistic Regression accuracy: {train_acc:.4f}")
