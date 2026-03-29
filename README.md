# EMI Tutorial 2 - Group 5
## King County Housing Price Prediction — MLOps Pipeline

This repository implements a fully orchestrated MLOps pipeline for predicting King County housing prices. It features automated data ingestion, preprocessing, model training, and evaluation, all managed through Apache Airflow, with configuration via Hydra and experiment tracking via MLflow.

## Project Structure
```
├── conf/config.yaml        # Hydra configuration (paths, hyperparameters, API settings)
├── dags/housing_pipeline.py # Airflow DAG definitions (2 DAGs: data stream + reactive training)
├── src/
│   ├── mock_api.py          # Simulated external API for housing data batches
│   ├── data_ingestion.py    # Data ingestion and validation
│   ├── preprocess.py        # Feature selection, cleaning, and train/test split
│   ├── train.py             # Random Forest model training
│   └── evaluate.py          # Model evaluation and unified MLflow logging
├── data/
│   └── raw/                 # DVC-tracked raw datasets (baseline + future batches)
├── docker-compose.yaml      # Full infrastructure: Airflow 3 + PostgreSQL + MLflow
├── Dockerfile               # Custom Airflow image with ML dependencies
└── requirements.txt         # Python dependencies
```

## Architecture

The system uses a **two-DAG event-driven architecture**:

1. **Data Stream DAG** (`data_stream_dag`) — Runs on a configurable schedule, simulates API calls to fetch new housing data, and updates the DVC-tracked baseline dataset.
2. **Reactive Training DAG** (`reactive_training_dag`) — Automatically triggered when the baseline dataset changes. Executes the full ML pipeline: **Ingestion → Preprocessing → Training → Evaluation**, logging all metrics, hyperparameters, and data lineage (DVC MD5 hash) to MLflow.

## How to Run

### Prerequisites
- Docker and Docker Compose installed

### 1. Launch the Infrastructure
```bash
docker compose up --build -d
```
This starts:
- **Airflow** (scheduler, webserver, DAG processor) at [http://localhost:8081](http://localhost:8081)
- **MLflow Tracking Server** at [http://localhost:5000](http://localhost:5000)
- **PostgreSQL** as Airflow's metadata database

### 2. Restore Data via DVC
```bash
dvc checkout
```

### 3. Monitor the Pipeline
- Open the **Airflow UI** at `http://localhost:8081` to see both DAGs
- The Data Stream DAG runs on schedule, triggering the Reactive Training DAG automatically
- Open the **MLflow UI** at `http://localhost:5000` to verify logged metrics, hyperparameters, model artifacts, and DVC data lineage tags

### 4. Modify Configuration (Hydra)
All pipeline parameters are centralized in `conf/config.yaml`:
- **File paths**: raw data, processed data, model output
- **Preprocessing**: test split ratio, random state
- **Training**: number of estimators, max depth
- **Mock API**: update frequency, data pool paths

Changes to the config are automatically picked up on the next pipeline run.

### 5. Shut Down
```bash
docker compose down
```