import os
from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from src.configuration.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import logging

config = ConfigurationManager()


class DataIngestion:
    def __init__(self, data_ingestion_config=config.get_data_ingestion_config()):
        self.data_ingestion_config = data_ingestion_config
        
        # Creating root dir for ingestion artifacts
        Path(self.data_ingestion_config.get("root_dir")).mkdir(parents=True, exist_ok=True)
        
    def get_engine(self):
        """
        Create and return a SQLAlchemy engine using credentials from the .env file.
        """
        load_dotenv() 

        try:
            pg_host = os.getenv("RDS_HOST")
            pg_user = os.getenv("RDS_USER")
            pg_password = os.getenv("RDS_PASSWORD")
            pg_db = os.getenv("RDS_DB")
            pg_port = os.getenv("RDS_PORT")
            
            print(f"host: {pg_host}, user: {pg_user}, pass: {pg_password}, db: {pg_db}, port: {pg_port}")

            if not all([pg_host, pg_user, pg_password, pg_db, pg_port]):
                logging.info("One or more environment variables for PostgreSQL database are missing.")
                raise EnvironmentError("Missing database connection parameters in .env file.")

            # Construct the SQLAlchemy connection URL
            DATABASE_URL = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
            engine = create_engine(DATABASE_URL)
            logging.info("SQLAlchemy engine created successfully.")
            
            return engine
        except Exception as e:
            raise CustomException(e, sys)



    def ingest_data_from_pg_db(self):
        """
        Fetch data from the given table using the provided SQLAlchemy engine and return as a DataFrame.
        """
        engine = self.get_engine()
        
        try:
            table_name = self.data_ingestion_config.get("database_table_name")
            logging.info(f"Executing query to fetch data from table {table_name}.")
            
            query = f"SELECT * FROM {table_name};"
            
            df = pd.read_sql_query(query, engine)
            logging.info(f"Loaded {len(df)} records from table.")
            
            return df
        except Exception as e:
            logging.error(f"Error fetching data from table {table_name}: {e}")
            raise CustomException(e, sys)




if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.ingest_data_from_pg_db()
