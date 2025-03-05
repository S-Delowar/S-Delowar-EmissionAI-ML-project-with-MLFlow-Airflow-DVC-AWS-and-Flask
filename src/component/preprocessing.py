import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv
import joblib
import mlflow
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.configuration.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import logging


config = ConfigurationManager()

class DataTransformation:
    def __init__(self, transformation_config=config.get_data_transformation_config()):
        self.transformation_config = transformation_config
        
        load_dotenv()
        
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("EmissionAI ML Project")
        
    def transform_data(self):
        try:
            df = pd.read_csv(self.transformation_config.get("input_data_path"))
            
            # drop unnecessary columns
            df.drop(columns=["id", "model_year", "make", "model", "co2_rating", "smog_rating"], axis=1, inplace=True)
            logging.info("Drop unnecessary columns")
            
            # Define features and target
            X = df.drop("co2_emissions_g_per_km", axis=1)
            y = df["co2_emissions_g_per_km"]

            # Get Numerical and Categorical Columns
            numerical_cols = X.select_dtypes(include=['number']).columns.to_list()
            categorical_cols = X.select_dtypes(exclude=['number']).columns.to_list()
            
            # Numeric columns transformer
            numerical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical column transformer
            categorical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            # Column transformer
            column_transformer = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )
            # Preprocessor pipeline
            preprocessor = Pipeline(steps=[
                ('column_transformer', column_transformer)
            ])
            logging.info("Preprocessor object created")
            
            logging.info(f"Splitted into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
            # Fit the preprocessor and transform train features
            X_train_transformed = preprocessor.fit_transform(X_train)
            # Transform test features
            X_test_transformed = preprocessor.transform(X_test)            
            logging.info(f"Preprocessor object fitted with train features and transformed train nd test features")
            
            # If X_train_transformed is a sparse matrix, convert it to a dense array
            if hasattr(X_train_transformed, "toarray"):
                X_train_transformed = X_train_transformed.toarray()

            
            # If X_test_transformed is a sparse matrix, convert it to a dense array
            if hasattr(X_test_transformed, "toarray"):
                X_test_transformed = X_test_transformed.toarray()

            # Check the shape and type of X_train_transformed
            print(f"Shape of X_train_transformed: {X_train_transformed.shape}")
            print(f"Type of X_train_transformed: {type(X_train_transformed)}")

            
            # Get the correct feature names after OneHotEncoding
            categorical_columns_after_encoding = preprocessor.named_steps['column_transformer'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols)
            all_column_names = list(numerical_cols) + list(categorical_columns_after_encoding)
            
            logging.info(f"Expected number of columns: {len(all_column_names)}")
            logging.info(f"Actual number of columns in transformed data: {X_train_transformed.shape[1]}")


            # Convert the preprocessed and resampled data back to DataFrames
            transformed_train_data = pd.DataFrame(X_train_transformed, columns=all_column_names)
            transformed_train_data['co2_emissions_g_per_km'] = y_train.values
            
            # Convert the preprocessed and resampled data back to DataFrames
            transformed_test_data = pd.DataFrame(X_test_transformed, columns=all_column_names)
            transformed_test_data['co2_emissions_g_per_km'] = y_test.values
            
            logging.info(f"Created transformed train and test sets")
            
            #Create directory for transformed data
            transformed_data_save_dir = self.transformation_config.get("transformed_data_dir")
            Path(transformed_data_save_dir).mkdir(parents=True, exist_ok=True)
            
            # Define file paths
            train_file_path = Path(transformed_data_save_dir) / "transformed_train_data.csv"
            test_file_path = Path(transformed_data_save_dir) / "transformed_test_data.csv"
            
            # Save DataFrames to CSV
            transformed_train_data.to_csv(train_file_path, index=False)
            transformed_test_data.to_csv(test_file_path, index=False)
            logging.info(f"Saved transformed train and test data to '{transformed_data_save_dir}'")
            
            
            # Save preprocessor object
            preprocessor_save_path = self.transformation_config.get("preprocessor_save_path")
            # Create parent directory
            Path(preprocessor_save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(preprocessor_save_path, 'wb') as file:
                joblib.dump(preprocessor, file)
            
            logging.info(f"Preprocessor object saved to '{preprocessor_save_path}'")
            
            # Logging preprocessor object to mlflow
            with mlflow.start_run(run_name="preprocessor-object", nested=True):
                # mlflow.log_artifact(preprocessor_save_path, artifact_path="preprocessor")
                tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                
                signature = infer_signature(X_train, X_train_transformed)
                
                if tracking_uri_type_store != "file":
                    mlflow.sklearn.log_model(
                        preprocessor, 
                        "preprocessor", 
                        signature=signature,
                        registered_model_name="emission_ai_preprocessor"
                    )
                else:
                    mlflow.sklearn.log_model(
                        preprocessor, 
                        "preprocessor",
                        signature=signature
                    )
                logging.info("Logged preprocessor object to MLflow")
              
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    data_transformation = DataTransformation()
    data_transformation.transform_data()    
