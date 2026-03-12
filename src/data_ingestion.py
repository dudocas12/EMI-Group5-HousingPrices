import pandas as pd
import mlflow
import yaml

def get_dvc_hash(dvc_file_path):
    """Reads the tiny DVC file to extract the MD5 hash of your dataset."""
    with open(dvc_file_path, 'r') as file:
        dvc_info = yaml.safe_load(file)
        # DVC files store the hash inside the 'outs' list
        return dvc_info['outs'][0]['md5']

def main():
    # 1. Define file paths (assuming you run this from the root folder)
    data_path = "data/raw/kc_house_data.csv"
    dvc_path = "data/raw/kc_house_data.csv.dvc"

    # 2. Ingest the Data
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with {len(df)} rows.")

    # 3. Get the exact DVC Hash
    dataset_hash = get_dvc_hash(dvc_path)
    print(f"Dataset DVC Hash: {dataset_hash}")

    # 4. Log everything to MLflow
    mlflow.set_experiment("Housing_Prices_Baseline")
    
    with mlflow.start_run():
        # This is the exact requirement: logging the MD5 hash
        mlflow.set_tag("dvc_md5_hash", dataset_hash)
        
        # We can also log some basic info just to show it works
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("num_columns", len(df.columns))
        
        print("Successfully logged data lineage to MLflow!")

if __name__ == "__main__":
    main()