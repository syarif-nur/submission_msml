import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Submission")

data = pd.read_csv("wine-qt_preprocessing.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("label", axis=1),
    data["label"],
    random_state=42,
    test_size=0.2
)
input_example = X_train.iloc[0:5]

with mlflow.start_run():
    # Log parameters
    n_estimators = 50
    max_depth = 5
    mlflow.autolog()
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)