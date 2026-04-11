"""
Runs a tournament between all trained models, selects the champion based on RMSE,
and registers the winning model in the MLflow Model Registry.
"""
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
    test_path = cfg.paths.test_data
    model_dir = cfg.paths.model_dir
    dvc_path = cfg.paths.dvc_tracker
    
    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=['price'])
    y_test = df_test['price']

    # Load all contenders
    print(f"Loading trained models from {model_dir}...")
    models = {
        "RandomForest": joblib.load(os.path.join(model_dir, "rf_model.pkl")),
        "LinearRegression": joblib.load(os.path.join(model_dir, "lr_model.pkl")),
        "XGBoost": joblib.load(os.path.join(model_dir, "xgb_model.pkl")),
    }

    # Evaluate each model
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "model": model}
        print(f"  {name:20s} -> RMSE: {rmse:>12,.2f}  MAE: {mae:>12,.2f}  R2: {r2:.4f}")

    # Determine the champion
    champion_name = min(results, key=lambda k: results[k]["rmse"])
    champion = results[champion_name]
    print(f"\nChampion: {champion_name} (RMSE: {champion['rmse']:,.2f})")

    # Extract DVC hash for data lineage
    with open(dvc_path, 'r') as file:
        dvc_info = yaml.safe_load(file)
        dataset_hash = dvc_info['outs'][0]['md5']

    # Log everything to MLflow
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run():
        # Data lineage
        mlflow.set_tag("dvc_md5_hash", dataset_hash)
        
        # Log all hyperparameters
        mlflow.log_param("rf_n_estimators", cfg.training.random_forest.n_estimators)
        mlflow.log_param("rf_max_depth", cfg.training.random_forest.max_depth)
        mlflow.log_param("xgb_n_estimators", cfg.training.xgboost.n_estimators)
        mlflow.log_param("xgb_max_depth", cfg.training.xgboost.max_depth)
        mlflow.log_param("xgb_learning_rate", cfg.training.xgboost.learning_rate)
        mlflow.log_param("random_state", cfg.training.random_state)

        # Log RMSE for every contender (visible in MLflow comparison charts)
        for name, res in results.items():
            mlflow.log_metric(f"{name}_rmse", res["rmse"])
            mlflow.log_metric(f"{name}_mae", res["mae"])
            mlflow.log_metric(f"{name}_r2", res["r2"])

        # Log the champion's identity and final stats
        mlflow.log_param("champion_algorithm", champion_name)
        mlflow.log_metric("champion_rmse", champion["rmse"])
        mlflow.log_metric("champion_mae", champion["mae"])
        mlflow.log_metric("champion_r2", champion["r2"])
        
        # Register only the winner in the Model Registry
        mlflow.sklearn.log_model(
            sk_model=champion["model"], 
            artifact_path="champion_model",
            registered_model_name="KingCounty_Champion"
        )
        
        print(f"Logged unified run to MLflow (Hash: {dataset_hash})")

if __name__ == "__main__":
    evaluate_models()