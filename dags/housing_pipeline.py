from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.datasets import Dataset
from datetime import datetime, timedelta
import yaml

# Read the shared Hydra config for pipeline-wide parameters
with open('/opt/airflow/conf/config.yaml', 'r') as f:
    pipeline_cfg = yaml.safe_load(f)

UPDATE_DAYS = pipeline_cfg['mock_api']['update_frequency_days']

default_args = {
    'owner': 'group5',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# The Dataset acts as the bridge
baseline_dataset = Dataset('file:///opt/airflow/data/raw/baseline.csv')

# DAG 1: The Scheduled Data Stream
with DAG(
    'data_stream_dag',
    default_args=default_args,
    description=f'Simulates API fetches every {UPDATE_DAYS} days',
    schedule=timedelta(days=UPDATE_DAYS),
    catchup=False,
    tags=['mlops', 'ingestion', 'api'],
) as stream_dag:

    # Step 1: Fetch the new data
    fetch_api_task = BashOperator(
        task_id='fetch_mock_api',
        bash_command='python src/mock_api.py',
        cwd='/opt/airflow',
    )

    # Step 2: Force DVC to track the new data
    update_dvc_task = BashOperator(
        task_id='update_dvc_hashes',
        bash_command='dvc add data/raw/baseline.csv data/raw/future_batches.csv',
        cwd='/opt/airflow',
        outlets=[baseline_dataset]
    )

    # Define the order
    fetch_api_task >> update_dvc_task

# DAG 2: The Reactive ML Pipeline
with DAG(
    'reactive_training_dag',
    default_args=default_args,
    description='Reacts to new baseline data: processes, trains, and evaluates',
    schedule=[baseline_dataset],  
    catchup=False,
    tags=['mlops', 'training', 'event-driven'],
) as training_dag:

    ingest_task = BashOperator(
        task_id='ingest_data',
        bash_command='python src/data_ingestion.py',
        cwd='/opt/airflow',
    )

    preprocess_task = BashOperator(
        task_id='preprocess_data',
        bash_command='python src/preprocess.py',
        cwd='/opt/airflow',
    )

    train_task = BashOperator(
        task_id='train_model',
        bash_command='python src/train.py',
        cwd='/opt/airflow',
    )

    evaluate_task = BashOperator(
        task_id='evaluate_model',
        bash_command='python src/evaluate.py',
        cwd='/opt/airflow',
    )

    ingest_task >> preprocess_task >> train_task >> evaluate_task