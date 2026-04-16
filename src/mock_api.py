"""
Mock API Module.
Simulates an external data source by releasing new housing records from a pool
of future batches on a configurable schedule, appending them to the baseline dataset.
"""
import pandas as pd
import os
from datetime import timedelta
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def fetch_new_batch(cfg: DictConfig):
    """Fetches the next chronological batch of housing data."""
    future_data_path = cfg.paths.future_batches
    baseline_path = cfg.paths.raw_baseline
    cursor_path = cfg.paths.cursor
    update_frequency_days = cfg.mock_api.update_frequency_days

    if not os.path.exists(future_data_path) or not os.path.exists(baseline_path):
        print("Error: Required data files are missing.")
        return

    df_future = pd.read_csv(future_data_path)
    df_baseline = pd.read_csv(baseline_path)

    if len(df_future) == 0:
        print("No more data available in the simulated API pool.")
        return

    df_future['date_parsed'] = pd.to_datetime(df_future['date'], format='%Y%m%dT%H%M%S', errors='coerce')

    # Read the simulation clock
    if os.path.exists(cursor_path):
        with open(cursor_path, 'r') as f:
            last_known_date = pd.to_datetime(f.read().strip())
    else:
        df_baseline['date_parsed'] = pd.to_datetime(df_baseline['date'], format='%Y%m%dT%H%M%S', errors='coerce')
        last_known_date = df_baseline['date_parsed'].max()
        df_baseline = df_baseline.drop(columns=['date_parsed'])

    # Advance the clock based on configured frequency
    target_end_date = last_known_date + timedelta(days=update_frequency_days)

    with open(cursor_path, 'w') as f:
        f.write(str(target_end_date))

    print(f"API Call: Fetching properties from {last_known_date.date()} to {target_end_date.date()} ({update_frequency_days} days)...")

    mask = df_future['date_parsed'] <= target_end_date
    df_batch = df_future[mask].copy()
    df_future = df_future[~mask].copy()

    if not df_batch.empty:
        df_batch = df_batch.drop(columns=['date_parsed'])
    df_future = df_future.drop(columns=['date_parsed'])

    if len(df_batch) > 0:
        updated_baseline = pd.concat([df_baseline, df_batch], ignore_index=True)
        updated_baseline.to_csv(baseline_path, index=False)
        df_future.to_csv(future_data_path, index=False)
        print(f"Fetched {len(df_batch)} new records and appended to baseline.")
    else:
        print(f"No new property sales found in this window. Clock advanced to {target_end_date.date()}.")

if __name__ == "__main__":
    fetch_new_batch()