import json
from pathlib import Path
import pandas as pd
from src.configuration.configuration import ConfigurationManager
from src.utils.logger import logging

config = ConfigurationManager()

class DataValidation:
    def __init__(self, validation_config=config.get_data_validation_config()):
        self.validation_config = validation_config
        
        # Create root directory for validation artifacts
        Path(self.validation_config.get("root_dir")).mkdir(parents=True, exist_ok=True)
        
    def get_validation_errors(self, df, schema):        
        errors = []
        
        schema_cols = schema.get("columns")
        
        for col in schema_cols:
            col_name = col.get("name")
            expected_type = col.get("type")
            
            if not col_name in df.columns:
                errors.append(f"Missing column: {col_name}")
                continue
            
            if expected_type=="int":
                if not pd.api.types.is_integer_dtype(df[col_name]):
                    errors.append(f"Column {col_name} is not of type int")
                
            elif expected_type=="float":
                if not pd.api.types.is_float_dtype(df[col_name]):
                    errors.append(f"Column {col_name} is not of type float")
                
            elif expected_type=="object":
                if not pd.api.types.is_object_dtype(df[col_name]):
                    errors.append(f"Column {col_name} is not of type object")
            else:
                errors.append(f"Unsupported expected type '{expected_type}' for column {col_name}")
                
        return errors
        
    def validate_data(self, df):
        schema = self.validation_config.get("validation_schema")
        validation_errors = self.get_validation_errors(df, schema)
        
        if validation_errors:
            errors_dict = {"errors": validation_errors}
            
            # Save errors
            errors_save_path = self.validation_config.get("errors_save_path")
            Path(errors_save_path).mkdir(parents=True, exist_ok=True)
            
            with open(errors_save_path, 'w') as file:
                json.dump(errors_dict, file)
                logging.info(f"Errors saved to '{errors_save_path}'")
            
            for error in validation_errors:
                logging.info(error)
        else:
            logging.info(f"Data Successfully Validated with Validation Schema")
            valid_data_save_path = self.validation_config.get("valid_data_save_path")
            Path(valid_data_save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(valid_data_save_path, index=False)
            logging.info(f"Valid Data Saved to '{valid_data_save_path}'")
            
            