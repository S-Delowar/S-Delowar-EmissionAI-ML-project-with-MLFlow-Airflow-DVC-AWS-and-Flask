from pathlib import Path
import subprocess
import sys
import os
from src.utils.exception import CustomException
from src.utils.logger import logging

from dotenv import load_dotenv

load_dotenv()
 
class VERSIONING:
    def __init__(self):
        try:
            # Check if DVC remote "origin" exists
            remotes = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True).stdout
            logging.info(f"DVC remote list: {remotes}")
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def version_artifacts(self):
        try:
            # Validate paths before DVC ops
            required_paths = [
                "artifacts/data_validation/valid_data/data.csv",
                "artifacts/data_transformation/transformed_data",
                "preprocessor/preprocessor.pkl",
                "model/best_model.pkl"
            ]
            
            for path in required_paths:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Missing required artifact: {path}")
            
            subprocess.run(["dvc", "add", "artifacts/data_validation/valid_data/data.csv"], check=True, capture_output=True, text=True)
            subprocess.run(["dvc", "add", "artifacts/data_transformation/transformed_data"], check=True)
            subprocess.run(["dvc", "add", "preprocessor/preprocessor.pkl"], check=True)
            subprocess.run(["dvc", "add", "model/best_model.pkl"], check=True)

            # # Push both DVC
            subprocess.run(["dvc", "pull", "-r", "origin"], check=True)
            subprocess.run(["dvc", "push", "-r", "origin"], check=True)
            
            logging.info(f"Data, Model and Preprocessor successfully versioned to S3 using DVC")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error syncing to s3: {e.stderr}")





if __name__=="__main__":
    versioning = VERSIONING()
    versioning.version_artifacts()