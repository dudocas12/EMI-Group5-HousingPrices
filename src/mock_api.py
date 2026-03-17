import pandas as pd
import os

def fetch_new_batch(batch_size=100):
    """Simulates calling an external API to get fresh housing data."""
    future_data_path = "data/raw/future_batches.csv"
    new_batch_path = "data/raw/new_api_data.csv"

    if not os.path.exists(future_data_path):
        print("Error: The API server (future_batches.csv) is offline or missing")
        return

    # "Connect" to the API and get the data
    df_future = pd.read_csv(future_data_path)

    if len(df_future) == 0:
        print("The API has no more new houses")
        return

    # Grab the top 100 oldest houses (chronologically)
    df_batch = df_future.head(batch_size)
    
    # Remove those 100 houses from the server to not get them again
    df_future = df_future.iloc[batch_size:]

    # Save the new batch locally and update the server
    df_batch.to_csv(new_batch_path, index=False)
    df_future.to_csv(future_data_path, index=False)

    print(f"API Success: Downloaded a fresh batch of {len(df_batch)} houses")
    print(f"Saved locally to: {new_batch_path}")
    print(f"Houses remaining on server: {len(df_future)}")

if __name__ == "__main__":
    fetch_new_batch()