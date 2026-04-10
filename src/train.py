import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_models(cfg: DictConfig):
    print(f"Loading training data from {cfg.paths.train_data}...")
    df_train = pd.read_csv(cfg.paths.train_data)

    # Separate features (X) and target (y)
    X_train = df_train.drop(columns=['price'])
    y_train = df_train['price']

    # Ensure the models directory exists
    os.makedirs(cfg.paths.model_dir, exist_ok=True)

    # Train the Champion: Random Forest
    print(f"Initializing RandomForestRegressor (Trees: {cfg.training.n_estimators}, Depth: {cfg.training.max_depth})...")
    rf_model = RandomForestRegressor(
        n_estimators=cfg.training.n_estimators, 
        max_depth=cfg.training.max_depth, 
        random_state=cfg.training.random_state
    )
    
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    rf_path = os.path.join(cfg.paths.model_dir, "rf_model.pkl")
    joblib.dump(rf_model, rf_path)
    print(f"Random Forest saved successfully to {rf_path}")

    # Train the Challenger: Linear Regression
    print("Initializing Linear Regression (Challenger)...")
    lr_model = LinearRegression()
    
    print("Training Linear Regression model...")
    lr_model.fit(X_train, y_train)
    
    lr_path = os.path.join(cfg.paths.model_dir, "lr_model.pkl")
    joblib.dump(lr_model, lr_path)
    print(f"Linear Regression saved successfully to {lr_path}")

    print("Tournament contenders are ready for evaluation.")

if __name__ == "__main__":
    train_models()