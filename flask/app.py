import os
from flask import request, Flask
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# Set the MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# List registered models

# for model in mlflow.search_registered_models():
#     print(model.name)

# Define the model names
model_name = "emission_ai_best_model"
preprocessor_name = "emission_ai_preprocessor"

# Load the latest version of the preprocessor and model
preprocessor = mlflow.sklearn.load_model(model_uri=f"models:/{preprocessor_name}/latest")
model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/latest")


app = Flask(__name__)


@app.route("/")
def index():
    return "Welcome to Emission AI apis"

# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    data_df = pd.DataFrame(data)
    transformed_data = preprocessor.transform(data_df)
    prediction_result = model.predict(transformed_data)
    
    return {"prediction": prediction_result.tolist()}
    
    

if __name__=="__main__":
    app.run(
        host="0.0.0.0", port=8000, debug=True
    )

