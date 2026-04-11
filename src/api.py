from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import shap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import datetime
import time

os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow_server:5000"

app = FastAPI(title="King County Housing Prices API")

MODEL_NAME = "KingCounty_Champion"

# Human-readable labels for each feature (used in the explainability response)
FEATURE_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "sqft_living": "Living Area",
    "sqft_lot": "Lot Size",
    "floors": "Floors",
    "waterfront": "Waterfront Access",
    "view": "View Quality",
    "condition": "Property Condition",
    "grade": "Construction Grade",
    "sqft_above": "Above-Ground Area",
    "sqft_basement": "Basement Area",
    "yr_built": "Year Built",
    "yr_renovated": "Renovation Status",
    "zipcode": "Neighborhood (Zip)",
    "lat": "North-South Location",
    "long": "East-West Location",
    "sqft_living15": "Neighbor Home Sizes",
    "sqft_lot15": "Neighbor Lot Sizes",
}

# Global state for the loaded model and its explainer
champion_algo = "Unknown"
champion_rmse = 0.0
model = None
explainer = None
background_data = None

def load_model_with_retry(max_retries=5, delay_seconds=10):
    """Loads the champion model from MLflow Registry with retry logic for resilience."""
    global champion_algo, champion_rmse, explainer, background_data
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}: Loading {MODEL_NAME}...")
            loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")

            # Retrieve metadata from the MLflow run that produced this model
            client = MlflowClient()
            # Get the latest version's run_id
            versions = client.get_latest_versions(MODEL_NAME)
            if versions:
                run_id = versions[0].run_id
                run = client.get_run(run_id)
                champion_algo = run.data.params.get("champion_algorithm", "Unknown Model")
                champion_rmse = float(run.data.metrics.get("champion_rmse", 0.0))

            print(f"Loaded {champion_algo} (RMSE: {champion_rmse})")

            # Check if the model supports SHAP TreeExplainer
            if hasattr(loaded_model, 'estimators_') or hasattr(loaded_model, 'get_booster'):
                explainer = "tree"
                # Load a sample of training data as SHAP background reference
                try:
                    train_df = pd.read_csv("/opt/airflow/data/processed/train.csv")
                    background_data = train_df.drop(columns=['price']).sample(n=100, random_state=42)
                    print(f"SHAP TreeExplainer ready with {len(background_data)} background samples.")
                except Exception as bg_err:
                    background_data = None
                    explainer = None
                    print(f"Could not load training data for SHAP: {bg_err}")
            else:
                explainer = None
                print("SHAP not available for this model type.")

            return loaded_model
        except Exception as e:
            print(f"Failed to load model: {e}")
            if attempt < max_retries:
                time.sleep(delay_seconds)
            else:
                return None

model = load_model_with_retry()

current_year = datetime.datetime.now().year

# Input validation schema with King County-specific constraints
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
    global model, champion_algo, champion_rmse, explainer

    if model is None:
        print("Model is missing. Attempting reload...")
        model = load_model_with_retry(max_retries=1)
        if model is None:
            raise HTTPException(status_code=503, detail="The model is currently training. Please try again in a few seconds.")
    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(df)
        predicted_price = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)

        # Compute SHAP values for explainability
        feature_contributions = []
        if explainer == "tree" and background_data is not None:
            try:
                shap_explainer = shap.Explainer(model.predict, background_data)
                shap_values = shap_explainer(df)
                values = shap_values.values[0]
                base_value = float(shap_values.base_values[0])
                feature_names = df.columns.tolist()

                for i, fname in enumerate(feature_names):
                    feature_contributions.append({
                        "feature": fname,
                        "label": FEATURE_LABELS.get(fname, fname),
                        "impact": round(float(values[i]), 2),
                    })

                feature_contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
            except Exception as e:
                print(f"SHAP computation failed (non-fatal): {e}")
                base_value = predicted_price
        else:
            base_value = predicted_price

        return {
            "predicted_price": predicted_price,
            "model_used": champion_algo,
            "rmse": champion_rmse,
            "base_value": round(base_value, 2),
            "feature_contributions": feature_contributions,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))