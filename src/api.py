from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import os
import datetime

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

current_year = datetime.datetime.now().year
# The "Bouncer": Enforcing strict data types for incoming prediction requests
class HousingInput(BaseModel):
    bedrooms: int = Field(ge=0, description="Cannot have negative bedrooms")
    bathrooms: float = Field(ge=0.0, description="Cannot have negative bathrooms")
    sqft_living: int = Field(gt=0, description="Living space must be greater than 0")
    sqft_lot: int = Field(gt=0)
    floors: float = Field(ge=1.0, description="Must have at least 1 floor")
    waterfront: int = Field(ge=0, le=1, description="Must be exactly 0 (No) or 1 (Yes)")
    view: int = Field(ge=0, le=4, description="View score must be between 0 and 4")
    condition: int = Field(ge=1, le=5, description="Condition score must be between 1 and 5")
    grade: int = Field(ge=1, le=13, description="Grade score must be between 1 and 13")
    sqft_above: int = Field(ge=0)
    sqft_basement: int = Field(ge=0)
    yr_built: int = Field(ge=1900, le=current_year, description=f"Must be a valid year up to {current_year}")
    yr_renovated: int = Field(ge=0, le=current_year, description=f"0 if never, or up to {current_year}")
    zipcode: int = Field(ge=98000, le=99000, description="Must be a valid King County zip")
    lat: float = Field(ge=47.0, le=48.0, description="Must be within King County latitude")
    long: float = Field(ge=-123.0, le=-121.0, description="Must be within King County longitude")
    sqft_living15: int = Field(gt=0)
    sqft_lot15: int = Field(gt=0)

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