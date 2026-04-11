"""
Trains three competing models (Random Forest, Linear Regression, XGBoost)
and saves them for the evaluation tournament.
"""
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
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

    # Model 1: Random Forest
    rf_cfg = cfg.training.random_forest
    print(f"Training Random Forest (Trees: {rf_cfg.n_estimators}, Depth: {rf_cfg.max_depth})...")
    rf_model = RandomForestRegressor(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        random_state=cfg.training.random_state
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, os.path.join(cfg.paths.model_dir, "rf_model.pkl"))
    print("Random Forest saved.")

    # Model 2: Linear Regression
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, os.path.join(cfg.paths.model_dir, "lr_model.pkl"))
    print("Linear Regression saved.")

    # Model 3: XGBoost
    xgb_cfg = cfg.training.xgboost
    print(f"Training XGBoost (Trees: {xgb_cfg.n_estimators}, Depth: {xgb_cfg.max_depth}, LR: {xgb_cfg.learning_rate})...")
    xgb_model = XGBRegressor(
        n_estimators=xgb_cfg.n_estimators,
        max_depth=xgb_cfg.max_depth,
        learning_rate=xgb_cfg.learning_rate,
        random_state=cfg.training.random_state,
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(cfg.paths.model_dir, "xgb_model.pkl"))
    print("XGBoost saved.")

    print("All tournament contenders are ready for evaluation.")

if __name__ == "__main__":
    train_models()