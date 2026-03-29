"""
Model Evaluation Module - Group 5
This script calculates performance metrics and serves as our master MLflow logger,
bundling data lineage, hyperparameters, metrics, and the model into a single run.
"""
import pandas as pd
import joblib
import mlflow
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def evaluate_model(cfg: DictConfig):
    # Load paths from Hydra
    test_path = cfg.paths.test_data
    model_path = cfg.paths.model_path
    dvc_path = cfg.paths.dvc_tracker
    
    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=['price'])
    y_test = df_test['price']

    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)

    # Generate Predictions
    print("Generating predictions on unseen test data...")
    predictions = model.predict(X_test)

    # Calculate Performance Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Evaluation Results ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R2:   {r2:.4f}\n")

    # NEW STRATEGY: Read the DVC file as raw text to find the hash
    print("Extracting DVC hash via raw text parsing...")
    with open(dvc_path, 'r') as file:
        raw_text = file.read()
        # This splits the text at "md5:", takes the second half, and grabs the first word (the hash)
        dataset_hash = raw_text.split('md5:')[1].split()[0].strip()

    # Log everything to MLflow in ONE unified run
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run():
        # 1. Strict Data Lineage
        mlflow.set_tag("dvc_md5_hash", dataset_hash)
        
        # 2. Model Hyperparameters
        mlflow.log_param("n_estimators", cfg.training.n_estimators)
        mlflow.log_param("max_depth", cfg.training.max_depth)
        mlflow.log_param("random_state", cfg.training.random_state)

        # 3. Performance Metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # 4. Model Artifact
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Successfully logged unified run to MLflow! (Hash: {dataset_hash})")

if __name__ == "__main__":
    evaluate_model()