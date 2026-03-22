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

def train_model():
    # 1. Load the preprocessed training data
    train_path = "data/processed/train.csv"
    print(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path)

    # Separate features (X) and target (y)
    X_train = df_train.drop(columns=['price'])
    y_train = df_train['price']

    # 2. Initialize the Model
    # NOTE: Hyperparameters like n_estimators and max_depth are currently hardcoded.
    # In the next step, Hydra will inject these dynamically from config.yaml!
    n_estimators = 100
    max_depth = 10
    random_state = 42
    
    print(f"Initializing RandomForestRegressor (Trees: {n_estimators}, Depth: {max_depth})...")
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )

    # 3. Train the Model
    print("Training model... (This might take a few seconds)")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Serialize and Save the Model
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest.pkl"
    joblib.dump(model, model_path)
    
    print(f"Trained model saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()