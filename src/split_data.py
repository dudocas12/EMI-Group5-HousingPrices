import pandas as pd

def split_dataset():
    # 1. Load your raw data
    data_path = "data/raw/kc_house_data.csv"
    print("Loading data...")
    df = pd.read_csv(data_path)

    # 2. Sort it chronologically
    df = df.sort_values(by='date')

    # 3. Split it (80% for baseline, 20% for the future)
    split_index = int(len(df) * 0.8)
    baseline_df = df.iloc[:split_index]
    future_df = df.iloc[split_index:]

    # 4. Save the two new files
    baseline_df.to_csv("data/raw/baseline.csv", index=False)
    future_df.to_csv("data/raw/future_batches.csv", index=False)
    
    print(f"Created baseline.csv with {len(baseline_df)} rows.")
    print(f"Created future_batches.csv with {len(future_df)} rows.")

if __name__ == "__main__":
    split_dataset()