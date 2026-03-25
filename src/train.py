"""
Model Training Module - Group 5
This script loads the preprocessed training dataset and trains our core
regression model (Random Forest) to predict King County housing prices.
The trained model is then serialized for later evaluation and serving.
"""
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_model(cfg: DictConfig):
    print(f"Loading training data from {cfg.paths.train_data}...")
    df_train = pd.read_csv(cfg.paths.train_data)

    # Separate features (X) and target (y)
    X_train = df_train.drop(columns=['price'])
    y_train = df_train['price']

    print(f"Initializing RandomForestRegressor (Trees: {cfg.training.n_estimators}, Depth: {cfg.training.max_depth})...")
    model = RandomForestRegressor(
        n_estimators=cfg.training.n_estimators, 
        max_depth=cfg.training.max_depth, 
        random_state=cfg.training.random_state
    )

    # Train the Model
    print("Training model... (This might take a few seconds)")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Serialize and Save the Model
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    model_path = cfg.paths.model_path
    joblib.dump(model, model_path)
    
    print(f"Trained model saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()