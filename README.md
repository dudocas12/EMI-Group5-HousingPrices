# EMI Final Delivery
## King County Housing Price Prediction — MLOps Pipeline

This repository implements a fully orchestrated MLOps pipeline for predicting King County housing prices. It features automated data ingestion, preprocessing, a multi-model training tournament, and a production-ready serving layer — all managed through Apache Airflow, with configuration via Hydra and experiment tracking via MLflow.

## Project Structure
```
├── conf/config.yaml          # Hydra configuration (paths, hyperparameters, API settings)
├── dags/housing_pipeline.py  # Airflow DAG definitions (2 event-driven DAGs)
├── src/
│   ├── mock_api.py           # Simulated external API for housing data batches
│   ├── data_ingestion.py     # Data ingestion and validation
│   ├── preprocess.py         # Feature selection, cleaning, and train/test split
│   ├── train.py              # Multi-model training (Random Forest, Linear Regression, XGBoost)
│   ├── evaluate.py           # Model tournament and MLflow Model Registry promotion
│   ├── api.py                # FastAPI serving layer with SHAP explainability
│   └── app.py                # Streamlit frontend with interactive map and price breakdown
├── data/
│   └── raw/                  # DVC-tracked raw datasets (baseline + future batches)
├── docker-compose.yaml       # Full infrastructure (7 services)
├── Dockerfile                # Custom Airflow image with ML dependencies
└── requirements.txt          # Python dependencies
```

## Architecture

The system uses a **two-DAG event-driven architecture** with a serving layer:

### Pipeline (Airflow)
1. **Data Stream DAG** (`data_stream_dag`) — Runs on a configurable schedule, simulates API calls to fetch new housing data, and updates the DVC-tracked baseline dataset.
2. **Reactive Training DAG** (`reactive_training_dag`) — Automatically triggered when the baseline dataset changes. Executes: **Ingestion → Preprocessing → Training → Evaluation**.

### Model Tournament (MLflow)
The training step trains three competing models (Random Forest, Linear Regression, XGBoost). The evaluation step compares all three on RMSE, MAE, and R², and registers only the winner in the **MLflow Model Registry** as `KingCounty_Champion`.

### Serving Layer (FastAPI + Streamlit)
- **FastAPI** loads the latest champion from the registry and exposes a `/predict` endpoint with Pydantic validation and SHAP-based explainability.
- **Streamlit** provides an interactive map for location selection, property detail inputs, price estimation with confidence intervals, and a feature impact breakdown powered by SHAP.

## How to Run

### Prerequisites
- Docker and Docker Compose installed

### 1. Launch the Infrastructure
```bash
docker compose up --build -d
```
This starts 7 services:
- **Airflow** (scheduler, webserver, DAG processor) at [http://localhost:8081](http://localhost:8081)
- **MLflow Tracking Server** at [http://localhost:5000](http://localhost:5000)
- **FastAPI** at [http://localhost:8000](http://localhost:8000)
- **Streamlit** at [http://localhost:8501](http://localhost:8501)
- **PostgreSQL** as Airflow's metadata database

### 2. Restore Data via DVC
```bash
dvc checkout
```

### 3. Use the Application
1. Open the **Streamlit UI** at `http://localhost:8501`
2. Click a location on the map or enter coordinates manually
3. Fill in property details (bedrooms, sqft, grade, etc.)
4. Click **Generate Valuation** to get:
   - Estimated market value with confidence interval
   - A feature impact breakdown showing what drives the price

### 4. Monitor the Pipeline
- **Airflow UI** (`http://localhost:8081`) — View both DAGs and their execution history
- **MLflow UI** (`http://localhost:5000`) — Compare model metrics, view the Model Registry, and inspect data lineage (DVC hashes)

### 5. Modify Configuration (Hydra)
All pipeline parameters are centralized in `conf/config.yaml`:
- **File paths**: raw data, processed data, model output
- **Preprocessing**: test split ratio, random state
- **Training**: Random Forest (n_estimators, max_depth), XGBoost (n_estimators, max_depth, learning_rate)
- **Mock API**: update frequency

Changes to the config are automatically picked up on the next pipeline run.

### 6. Shut Down
```bash
docker compose down
```