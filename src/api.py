from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

# Set MLflow tracking URI so the API knows where to find the registry
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow_server:5000"

app = FastAPI(title="King County Housing Prices API")

# Load the latest registered model from MLflow
MODEL_NAME = "KingCounty_RandomForest"
print(f"Loading latest version of {MODEL_NAME} from MLflow...")
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# The "Bouncer": Enforcing strict data types for incoming prediction requests
class HousingInput(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float

@app.get("/")
def health_check():
    return {"status": "API is live", "model_loaded": model is not None}

@app.post("/predict")
def predict_price(data: HousingInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Convert the strictly validated Pydantic object into a pandas DataFrame
        df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(df)
        
        return {
            "predicted_price": float(prediction),
            "model_version": "latest"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))