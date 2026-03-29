# Engineering of Intelligent Models: Tutorial #2
**Group 5** | Configuration Management and Pipeline Orchestration

## Project Overview
This repository contains the Tutorial #2 submission for our MLOps pipeline, predicting King County housing prices. For this stage, we transitioned our experimental machine learning code into a modular, automated, and production-ready architecture. 

Our system completely eliminates manual execution steps and hardcoded parameters by orchestrating the workflow with Apache Airflow and managing configurations hierarchically via Hydra.

## Key MLOps Implementations

### 1. Configuration Management (Hydra)
All hardcoded variables have been extracted from our Python scripts. We utilize Hydra to dynamically inject file paths, MLflow settings, and Random Forest hyperparameters (`n_estimators`, `max_depth`, `random_state`) via a centralized `config.yaml` file. This ensures pure separation of source code from configuration.

### 2. Event-Driven Orchestration (Apache Airflow)
Our workflow is orchestrated using Apache Airflow into a Directed Acyclic Graph (DAG). To build a modern, reactive system, we separated our pipeline into two distinct DAGs:
* `data_stream_dag`: Handles data ingestion and DVC tracker updates.
* `reactive_training_dag`: An event-driven DAG that automatically triggers the workflow (Data Ingestion → Preprocessing → Training → Evaluation) the moment a new dataset update is detected by Airflow's Dataset scheduling.

### 3. Strict Data Lineage & Experiment Tracking (DVC + MLflow)
To maintain 100% reproducibility and adhere to stateless Airflow best practices, our final `evaluate.py` script reads the `.dvc` tracker file directly. It dynamically parses the raw text to extract the exact dataset MD5 hash. 

This allows us to log a single, unified experiment run in MLflow that binds together:
* The exact dataset DVC MD5 hash (Data Lineage)
* The model hyperparameters
* The performance metrics (RMSE, MAE, R2)
* The serialized `.pkl` model artifact

## Repository Structure
 EMI-05_Tutorial2
 ┣ conf                 # Hydra YAML configuration files
 ┣ dags                 # Airflow DAG definitions (stream & reactive)
 ┣ data                 
 ┃ ┣ raw                # .dvc tracker files (Data pulled via DVC)
 ┃ ┗ processed          # .dvc tracker files (Data pulled via DVC)
 ┣ screenshots          # Visual evidence of DAG execution and MLflow lineage
 ┣ src                  # Modularized Python scripts
 ┃ ┣ data_ingestion.py
 ┃ ┣ preprocess.py
 ┃ ┣ train.py
 ┃ ┗ evaluate.py
 ┣ docker-compose.yaml  # Infrastructure definition (live-synced volumes)
 ┗ README.md

## How to Run the Pipeline
1. **Retrieve the Dataset:** Run `dvc pull` to reconstruct the raw and processed `.csv` files from the local DVC cache.
2. **Boot the Infrastructure:** Run `docker compose up -d` to spin up the Airflow and MLflow containers.
3. **Trigger the Stream:** Navigate to the Airflow UI (`localhost:8081`) and manually trigger the `data_stream_dag`. 
4. **Observe Orchestration:** Watch the `reactive_training_dag` automatically trigger in response to the dataset update.
5. **Verify Lineage:** Navigate to the MLflow UI (`localhost:5000`) to view the consolidated `Housing_Prices_Baseline` experiment run, containing the metrics, model, and DVC hash.