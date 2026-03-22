"""
Model Evaluation Module - Group 5
This script loads our trained Random Forest model and the isolated testing dataset.
It generates predictions, calculates core performance metrics, and logs 
both the metrics and the serialized model to MLflow for our system of record.
"""
import pandas as pd
import joblib
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model():
    # 1. Load the test data and the trained model
    test_path = "data/processed/test.csv"
    model_path = "models/random_forest.pkl"
    
    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=['price'])
    y_test = df_test['price']

    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)

    # 2. Generate Predictions
    print("Generating predictions on unseen test data...")
    predictions = model.predict(X_test)

    # 3. Calculate Performance Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Evaluation Results ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAE (Mean Absolute Error):      {mae:.2f}")
    print(f"R2 Score:                       {r2:.4f}\n")

    # 4. Log everything to MLflow
    mlflow.set_experiment("Housing_Prices_Baseline")
    
    with mlflow.start_run():
        # Log the calculated metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log the actual model artifact to MLflow's registry system
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print("Successfully logged evaluation metrics and model artifact to MLflow.")

if __name__ == "__main__":
    evaluate_model()