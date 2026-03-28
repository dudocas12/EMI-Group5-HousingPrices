from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.datasets import Dataset
from datetime import datetime, timedelta

default_args = {
    'owner': 'group5',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

UPDATE_DAYS = 1

# The Dataset acts as the bridge
baseline_dataset = Dataset('file:///opt/airflow/data/raw/baseline.csv')

# ==========================================
# DAG 1: The Scheduled Data Stream
# ==========================================
with DAG(
    'group5_data_stream_dag',
    default_args=default_args,
    description=f'Simulates API fetches every {UPDATE_DAYS} days',
    schedule=timedelta(days=UPDATE_DAYS),
    catchup=False,
    tags=['mlops', 'ingestion', 'api'],
) as stream_dag:

    # Step 1: Fetch the new data
    fetch_api_task = BashOperator(
        task_id='fetch_mock_api',
        bash_command=f'python src/mock_api.py --days {UPDATE_DAYS}',
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

# ==========================================
# DAG 2: The Reactive ML Pipeline
# ==========================================
with DAG(
    'group5_reactive_training_dag',
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