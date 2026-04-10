import streamlit as st
import requests
import datetime
current_year = datetime.datetime.now().year

# The internal Docker network address for your FastAPI backend
API_URL = "http://fastapi:8000/predict"

st.set_page_config(page_title="King County Housing Predictor", layout="centered")

st.title("King County Housing Price Predictor")
st.markdown("Adjust the features below to predict the house price using our latest MLflow-registered Random Forest model.")

st.header("House Features")

# Organizing inputs into columns for a cleaner UI
col1, col2, col3 = st.columns(3)

with col1:
    # Changed to strict integers (no .0)
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
    sqft_living = st.number_input("Sqft Living", min_value=0, value=2000, step=100)
    view = st.number_input("View (0-4)", min_value=0, max_value=4, value=0, step=1)
    sqft_above = st.number_input("Sqft Above", min_value=0, value=1500, step=100)
    yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, value=0, step=1)
    sqft_living15 = st.number_input("Sqft Living (Neighbors)", min_value=0, value=1800, step=100)

with col2:
    # Bathrooms remains float for half-baths
    bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.25)
    
    # Changed to strict integers
    sqft_lot = st.number_input("Sqft Lot", min_value=0, value=5000, step=100)
    condition = st.number_input("Condition (1-5)", min_value=1, max_value=5, value=3, step=1)
    sqft_basement = st.number_input("Sqft Basement", min_value=0, value=500, step=100)
    zipcode = st.number_input("Zipcode", min_value=98000, max_value=99000, value=98103, step=1)
    sqft_lot15 = st.number_input("Sqft Lot (Neighbors)", min_value=0, value=5000, step=100)

with col3:
    # Floors remains float for half-stories
    floors = st.number_input("Floors", min_value=1.0, value=1.0, step=0.5)
    
    # FIXED: Added the '0' back to options so users can say 'No'
    waterfront = st.selectbox("Waterfront", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Changed to strict integers
    grade = st.number_input("Grade (1-13)", min_value=1, max_value=13, value=7, step=1)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=current_year, value=1990, step=1)
    
    # Lat/Long remain floats
    lat = st.number_input("Latitude", value=47.5112, format="%.4f")
    long = st.number_input("Longitude", value=-122.257, format="%.4f")

if st.button("Predict Price", type="primary"):
    # Package the inputs to match the FastAPI Pydantic schema exactly
    payload = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
        "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront,
        "view": view, "condition": condition, "grade": grade,
        "sqft_above": sqft_above, "sqft_basement": sqft_basement, "yr_built": yr_built,
        "yr_renovated": yr_renovated, "zipcode": zipcode, "lat": lat,
        "long": long, "sqft_living15": sqft_living15, "sqft_lot15": sqft_lot15
    }
    
    with st.spinner("Consulting the model..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            
            price = result["predicted_price"]
            version = result["model_version"]
            
            st.success(f"### Estimated Price: ${price:,.2f}")
            st.caption(f"Model version used: {version}")
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the FastAPI backend: {e}")