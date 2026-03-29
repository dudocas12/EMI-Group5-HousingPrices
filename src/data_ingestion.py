import pandas as pd
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # File paths loaded dynamically via Hydra
    data_path = cfg.paths.raw_baseline

    # Ingestion Validation
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")
    print("Data ingestion task complete. Passing baton to preprocessing...")

if __name__ == "__main__":
    main()