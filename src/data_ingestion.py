import pandas as pd
import mlflow
import yaml

def get_dvc_hash(dvc_file_path):
    """Reads the DVC file to extract the MD5 hash of the dataset."""
    with open(dvc_file_path, 'r') as file:
        dvc_info = yaml.safe_load(file)
        return dvc_info['outs'][0]['md5']

def main():
    # File paths
    data_path = "data/raw/baseline.csv"
    dvc_path = "data/raw/baseline.csv.dvc"

    # Ingestion of the Data
    print("Loading dataset")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with {len(df)} rows")

    # Get the exact DVC Hash
    dataset_hash = get_dvc_hash(dvc_path)
    print(f"Dataset DVC Hash: {dataset_hash}")

    # Logging everything to MLflow
    mlflow.set_experiment("Housing_Prices_Baseline")
    
    with mlflow.start_run():
        # Logging the MD5 hash
        mlflow.set_tag("dvc_md5_hash", dataset_hash)
        
        # Logging basic info
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("num_columns", len(df.columns))
        
        print("Successfully logged data lineage to MLflow")

if __name__ == "__main__":
    main()