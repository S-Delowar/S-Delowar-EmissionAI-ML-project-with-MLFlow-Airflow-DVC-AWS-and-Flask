import os
from pathlib import Path
from urllib.parse import urlparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from dotenv import load_dotenv

from src.configuration.configuration import ConfigurationManager
from src.utils.logger import logging

config = ConfigurationManager()

class ModelTrainer:
    def __init__(self, trainer_config = config.get_model_trainer_config()):
        self.trainer_config = trainer_config
        
        # Create directory for model trainer artifacts
        Path(self.trainer_config.get("root_dir")).mkdir(parents=True, exist_ok=True)
        
        load_dotenv()
        
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("EmissionAI ML Project")
    
    def hyperparameter_tuning(self, models, params, train_x, train_y, test_x, test_y):
        """Perform hyperparameter tuning for regression models"""
        tuning_report = []
        best_model = None
        best_score = float('inf')
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                logging.info(f"Tuning model {model_name}.....")
                param_grid = params.get("model_name", {})
                
                reg = GridSearchCV(
                    estimator= model, param_grid= param_grid, n_jobs=-1, verbose=2
                )            
                reg.fit(train_x, train_y)
                
                # Store results
                model_results = {
                    "model": model_name,
                    "best_params": reg.best_params_,
                    "best_mse": -reg.best_score_,
                    "best_rmse": np.sqrt(-reg.best_score_)
                }
                
                tuning_report.append(model_results)
                
                # Evaluate on test data
                y_pred = reg.predict(test_x)
                
                # Calculate regression metrics
                test_mse = mean_squared_error(test_y, y_pred)
                test_rmse = np.sqrt(test_mse)
                test_mae = mean_absolute_error(test_y, y_pred)
                test_r2_score = r2_score(test_y, y_pred)

                if -reg.best_score_ < best_score:
                    best_score = -reg.best_score_
                    best_model = reg.best_estimator_
                
                # MLflow logging
                mlflow.log_params(param_grid)
                mlflow.log_metrics({
                    'best_mse': -reg.best_score_,
                    'best_rmse': np.sqrt(-reg.best_score_),
                    'test_mse': test_mse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2_score
                })
            
        return tuning_report, best_model, best_score
    
    
    def train_and_evaluate(self):
        train_data = pd.read_csv(self.trainer_config.get("train_data_path"))
        test_data = pd.read_csv(self.trainer_config.get("test_data_path"))
        
        train_x = train_data.drop("co2_emissions_g_per_km", axis=1)
        train_y = train_data["co2_emissions_g_per_km"]
        
        test_x = test_data.drop("co2_emissions_g_per_km", axis=1)
        test_y = test_data["co2_emissions_g_per_km"]
        
        # define models
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(),
            "svr": SVR(),
            "knn": KNeighborsRegressor()
        }
        
        params = self.trainer_config.get("model_params")
        
        
        tuning_report, best_model, best_score = self.hyperparameter_tuning(
                                                    models=models, params=params, 
                                                    train_x=train_x, train_y=train_y, 
                                                    test_x=test_x, test_y=test_y
                                                )
        
        # Save tuning results
        tuning_report_save_path = self.trainer_config.get("hyperparameter_tuning_report_save_path")
        Path(tuning_report_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        tuning_report_df = pd.DataFrame(tuning_report)
        tuning_report_df.to_csv(tuning_report_save_path, index=False)
        
        if best_model:
            # Save model
            model_save_path = self.trainer_config.get("model_save_path")
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            
            if Path(model_save_path).exists():
                logging.info(f"model save path created")
            
            logging.info(f"Saving model to '{model_save_path}'")
            
            with open(model_save_path, 'wb') as file:
                joblib.dump(best_model, file)
                logging.info(f"Model Saved !")
            
            # MLFlow logging for the best model
            with mlflow.start_run(run_name="Best_Model", nested=True):
                mlflow.log_artifact(tuning_report_save_path, "hyperparameter_tuning_report")
                mlflow.log_params(best_model.get_params())
                mlflow.log_metrics({
                    "best_mse": best_score,
                    "best_rmse": np.sqrt(best_score)
                })
                
                signature = infer_signature(train_x, best_model.predict(test_x))
                
                tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                
                if tracking_uri_type_store != "file":
                    mlflow.sklearn.log_model(
                        best_model, 
                        "best_model", 
                        signature=signature,
                        registered_model_name="emission_ai_best_model"
                    )
                else:
                    mlflow.sklearn.log_model(
                        best_model, 
                        "best_model", 
                        signature=signature
                    )
                logging.info("Best model logged to MLflow")
                
            
                
if __name__ =="__main__":
    model_trainer = ModelTrainer()
    model_trainer.train_and_evaluate()