from pathlib import Path
from src.utils.common import read_yml

CONFIG_FILEPATH = "config/config.yml"
SCHEMA_FILEPATH = "config/schema.yml"
PARAMS_FILEPATH = "config/model_params.yml"


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILEPATH, schema_filepath=SCHEMA_FILEPATH, params_filepath=PARAMS_FILEPATH):
        self.config = read_yml(config_filepath)
        self.schema = read_yml(schema_filepath)
        self.params = read_yml(params_filepath)
                
        # Create root artifact directory
        artifacts_root = self.config.get("artifacts_root", "artifacts")
        Path(artifacts_root).mkdir(parents=True, exist_ok=True)
    
    
    def get_data_ingestion_config(self):
        ingestion_config = self.config.get("data_ingestion")
        
        return {
            "root_dir": Path(ingestion_config.get("root_dir")), 
            "database_table_name": str(ingestion_config.get("database_table_name")),
            }
    
        
    def get_data_validation_config(self):
        validation_config = self.config.get("data_validation")
        validation_schema = self.schema
        return {
            "root_dir": Path(validation_config.get("root_dir")),
            "errors_save_path": Path(validation_config.get("errors_save_path")),
            "valid_data_save_path": Path(validation_config.get("valid_data_save_path")),
            "validation_schema": dict(validation_schema)
        }
        
    
    def get_data_transformation_config(self):
        transformation_config = self.config.get("data_transformation")
        return {
            "root_dir": Path(transformation_config.get("root_dir")),
            "input_data_path": Path(transformation_config.get("input_data_path")),
            "preprocessor_save_path": Path(transformation_config.get("preprocessor_save_path")),
            "transformed_data_dir": Path(transformation_config.get("transformed_data_dir")),
        }
        
    def get_model_trainer_config(self):
        trainer_config = self.config.get("model_trainer")
        model_params = self.params        
        return {
            "root_dir": Path(trainer_config.get("root_dir")),
            "train_data_path": Path(trainer_config.get("train_data_path")),
            "test_data_path": Path(trainer_config.get("test_data_path")),
            "model_save_path": Path(trainer_config.get("model_save_path")),
            "hyperparameter_tuning_report_save_path": Path(trainer_config.get("hyperparameter_tuning_report_save_path")),
            "model_params": dict(model_params)
        }