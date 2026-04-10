import pandas as pd
import os
import joblib
import mlflow
import yaml
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def evaluate_models(cfg: DictConfig):
    # Load paths from Hydra
    test_path = cfg.paths.test_data
    model_dir = cfg.paths.model_dir  
    dvc_path = cfg.paths.dvc_tracker
    
    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=['price'])
    y_test = df_test['price']

    print(f"Loading trained models from {model_dir}...")
    rf_model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    lr_model = joblib.load(os.path.join(model_dir, "lr_model.pkl"))

    # Generate Predictions
    print("Generating predictions on unseen test data...")
    rf_predictions = rf_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)

    # Calculate Performance Metrics (Using RMSE as our primary tournament metric)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))

    print("\n Tournament Results \n")
    print(f"Random Forest RMSE:     {rf_rmse:.2f}")
    print(f"Linear Regression RMSE: {lr_rmse:.2f}\n")

    # Extract the DVC MD5 hash for strict data lineage tracking
    with open(dvc_path, 'r') as file:
        dvc_info = yaml.safe_load(file)
        dataset_hash = dvc_info['outs'][0]['md5']

    # Log everything to MLflow in ONE unified run
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run():
        # Strict Data Lineage
        mlflow.set_tag("dvc_md5_hash", dataset_hash)
        
        # Log Hyperparameters (Random Forest)
        mlflow.log_param("n_estimators", cfg.training.n_estimators)
        mlflow.log_param("max_depth", cfg.training.max_depth)
        mlflow.log_param("random_state", cfg.training.random_state)

        # Log both metrics so you can compare them in the UI
        mlflow.log_metric("rf_rmse", rf_rmse)
        mlflow.log_metric("lr_rmse", lr_rmse)
        
        # --- THE SHOWDOWN ---
        if rf_rmse < lr_rmse:
            print("Random Forest wins! Registering as Champion...")
            best_model = rf_model
            champion_name = "RandomForest"
            champion_rmse = rf_rmse
            champion_mae = mean_absolute_error(y_test, rf_predictions)
            champion_r2 = r2_score(y_test, rf_predictions)
        else:
            print("Linear Regression wins! Registering as Champion...")
            best_model = lr_model
            champion_name = "LinearRegression"
            champion_rmse = lr_rmse
            champion_mae = mean_absolute_error(y_test, lr_predictions)
            champion_r2 = r2_score(y_test, lr_predictions)

        # Log the winner's identity and final stats
        mlflow.log_param("champion_algorithm", champion_name)
        mlflow.log_metric("champion_rmse", champion_rmse)
        mlflow.log_metric("champion_mae", champion_mae)
        mlflow.log_metric("champion_r2", champion_r2)
        
        # Model Artifact & Registry (Only the winner gets registered!)
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="champion_model",
            registered_model_name="KingCounty_Champion"  # This is what FastAPI will look for
        )
        
        print(f"Successfully logged unified run to MLflow (Hash: {dataset_hash})")

if __name__ == "__main__":
    evaluate_models()