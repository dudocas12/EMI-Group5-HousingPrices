"""
Data Preprocessing Module - Group 5
This script cleans the raw King County housing baseline data, selects 
relevant predictive features, and splits the dataset into training 
and testing sets for model evaluation.
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def preprocess_data(cfg: DictConfig):
    # Load data using hydra config
    input_path = cfg.paths.raw_baseline
    print(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)

    # Feature Selection & Cleaning
    # Dropping 'id' and 'date' as they aren't numerical predictors for our baseline model.
    # We will predict 'price' using the remaining physical characteristics.
    if 'id' in df.columns and 'date' in df.columns:
        df = df.drop(columns=['id', 'date'])
        
    # Drop any rows with missing values to ensure a clean training set
    df = df.dropna()
    print(f"Data cleaned. Remaining records: {len(df)}")

    # Define Features (X) and Target (y)
    X = df.drop(columns=['price'])
    y = df['price']

    # Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.preprocessing.test_size, random_state=cfg.preprocessing.random_state)

    # Recombine to save as clean CSVs
    train_df = X_train.copy()
    train_df['price'] = y_train
    
    test_df = X_test.copy()
    test_df['price'] = y_test

    # Save the processed datasets
    os.makedirs(os.path.dirname(cfg.paths.train_data), exist_ok=True)
    train_path = cfg.paths.train_data
    test_path = cfg.paths.test_data
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Preprocessing complete")
    print(f"Training set saved to {train_path} ({len(train_df)} rows)")
    print(f"Testing set saved to {test_path} ({len(test_df)} rows)")

if __name__ == "__main__":
    preprocess_data()