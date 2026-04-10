from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import os
import datetime
import time

os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow_server:5000"

app = FastAPI(title="King County Housing Prices API")

MODEL_NAME = "KingCounty_Champion"

# The Resilience Loop
def load_model_with_retry(max_retries=5, delay_seconds=10):
    """Attempts to load the model multiple times before giving up."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}: Loading latest version of {MODEL_NAME}...")
            loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
            print("Model loaded successfully!")
            return loaded_model
        except Exception as e:
            print(f"Failed to load model: {e}")
            if attempt < max_retries:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print("Max retries reached. API will boot, but predictions are disabled.")
                return None

# Execute the loop when the API starts
model = load_model_with_retry()

current_year = datetime.datetime.now().year

# The Bouncer: Enforcing strict data types for incoming prediction requests
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
    global model
    # The Safety Net
    if model is None:
        print("Model is missing. Attempting a quick reload...")
        # Try one last time to grab it dynamically
        model = load_model_with_retry(max_retries=1)
        
        # If it is still missing, return a polite 503 instead of a fatal 500
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail="The model is currently training or initializing. Please try again in a few seconds."
            )
    
    try:
        # Convert the strictly validated Pydantic object into a pandas DataFrame
        df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(df)
        
        return {
            "predicted_price": float(prediction if isinstance(prediction, (list, pd.Series, pd.DataFrame)) else prediction),
            "model_version": "latest"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))