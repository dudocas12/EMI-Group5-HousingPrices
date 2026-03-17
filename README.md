# EMI Tutorial 1 - Group 5

This repository contains the foundational MLOps infrastructure for predicting King County housing prices. It implements data versioning (DVC), experiment tracking and data lineage (MLflow), and a simulated data pipeline for future automation.

## Project Structure
* **`data/raw/`**: Contains our active baseline dataset. 
* **`src/`**: Contains the data ingestion script, dataset splitting logic, and a mock API to simulate incoming data batches.
* **`.dvc/cache/`**: Serves as our local remote storage for DVC.

## How to Run This Project

### 1. Install Dependencies
Ensure you have a virtual environment active, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Restore the Data (DVC)
The raw dataset is intentionally excluded from the folder to demonstrate versioning. To restore the `baseline.csv` file from the local DVC cache, run:
```bash
dvc checkout
```

### 3. Run the Data Pipeline
Execute the baseline data ingestion script. This will process the data, read the DVC hash, and log the data lineage:
```bash
python src/data_ingestion.py
```

### 4. Verify Data Lineage (MLflow)
To verify that the strict data lineage (DVC MD5 hash) was successfully captured, boot up the MLflow UI:
```bash
mlflow ui
```
Navigate to `http://127.0.0.1:5000`, open the `Housing_Prices_Baseline` experiment on the left sidebar, and click on the most recent run. Check the **Tags** section to find the logged `dvc_md5_hash`.

### 5. (Optional) Simulate Future API Batches
To see the simulated data feed in action (preparing for automated orchestration in Tutorial #2), you can run our Mock API script. It fetches the next 100 chronological rows from our hidden server file and saves them locally:
```bash
python src/mock_api.py
```