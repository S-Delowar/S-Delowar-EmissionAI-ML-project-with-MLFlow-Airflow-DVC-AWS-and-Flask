from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task
from datetime import datetime, timedelta
import sys
import os
import pandas as pd

from src.component.data_ingestion import DataIngestion
from src.component.data_validation import DataValidation
import mlflow

# from src.component.model_trainer_and_evaluate import ModelTrainer
from src.component.preprocessing import DataTransformation
# from src.component.versioning import VERSIONING


# Define default_args
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 2, 17),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}



with DAG(
    "emissionAI_ml_pipeline",
    default_args=default_args,
    description="ML pipeline to predict CO2 emissions from vehicles",
    schedule_interval="@daily",
    catchup=False,
) as dag:
    
    @task
    def ingest_data():
        data_ingestion = DataIngestion()
        df = data_ingestion.ingest_data_from_pg_db()
        
        return df.to_json()   
       
    @task
    def validation(df_json:str):
        df = pd.read_json(df_json)
        
        data_validation = DataValidation()
        data_validation.validate_data(df)
        
    
    @task
    def preprocessing():
        data_transformer = DataTransformation()
        data_transformer.transform_data()
        
        
    # @task
    # def model_training_and_evaluating():
    #     model_trainer = ModelTrainer()
    #     model_trainer.train_and_evaluate()
    
    # @task
    # def data_and_model_versioning_dvc_s3():
    #     versioning = VERSIONING()
    #     versioning.version_artifacts()
        
          
    
    # dag dependencies
    validation(ingest_data()) >> preprocessing()
    # validation(ingest_data()) >> preprocessing() >> model_training_and_evaluating() >> data_and_model_versioning_dvc_s3()
