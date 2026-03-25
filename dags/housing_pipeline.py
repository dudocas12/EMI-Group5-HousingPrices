from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'group5',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'group5_housing_pipeline',
    default_args=default_args,
    description='End-to-end MLOps pipeline for King County housing prices',
    schedule='0 3 26 3 *',  # <-- This is the Airflow 3 specific fix
    catchup=False,
    tags=['mlops', 'baseline'],
) as dag:

    # Task 1: Data Ingestion
    ingest_task = BashOperator(
        task_id='ingest_data',
        bash_command='python src/data_ingestion.py',
        cwd='/opt/airflow',
    )

    # Task 2: Preprocessing
    preprocess_task = BashOperator(
        task_id='preprocess_data',
        bash_command='python src/preprocess.py',
        cwd='/opt/airflow',
    )

    # Task 3: Model Training
    train_task = BashOperator(
        task_id='train_model',
        bash_command='python src/train.py',
        cwd='/opt/airflow',
    )

    # Task 4: Model Evaluation
    evaluate_task = BashOperator(
        task_id='evaluate_model',
        bash_command='python src/evaluate.py',
        cwd='/opt/airflow',
    )

    # Define the strict chronological execution order
    ingest_task >> preprocess_task >> train_task >> evaluate_task