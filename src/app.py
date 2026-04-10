import streamlit as st
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium

API_URL = "http://fastapi:8000/predict"
CURRENT_YEAR = datetime.now().year

st.set_page_config(page_title="King County Predictor", page_icon="🏡", layout="wide")

# --- CACHED DATA FETCHING (For the Map Outline) ---
@st.cache_data
def get_king_county_geojson():
    """Fetches the official King County boundary so non-locals know where to click."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    try:
        data = requests.get(url).json()
        # Filter the massive US dataset for ONLY Washington (53) and King County (033)
        king_county = [f for f in data["features"] if f["id"] == "53033"]
        if king_county:
            return {"type": "FeatureCollection", "features": king_county}
    except Exception as e:
        print(f"Could not load county boundaries: {e}")
    return None

# Initialize session state for memory and notifications
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "show_success_toast" not in st.session_state:
    st.session_state.show_success_toast = False

# Trigger the sleek notification if a prediction just finished
if st.session_state.show_success_toast:
    st.toast("🎉 Valuation complete! The dashboard has been updated.", icon="✅")
    st.session_state.show_success_toast = False # Reset it so it doesn't spam the user

# --- HEADER ---
st.title("🏡 King County Housing Price Predictor")
st.markdown("Enter property details below and let our machine learning pipeline estimate the fair market value.")
st.divider()

# --- TOP ROW: DASHBOARD ---
top_col1, top_col2 = st.columns([1.5, 1])

with top_col1:
    st.subheader("📍 Location")
    # Tweak 1: Brought back the manual coordinate toggle
    location_mode = st.radio("Entry Method:", ["Interactive Map", "Manual Entry"], horizontal=True)
    
    if location_mode == "Interactive Map":
        south_west = [47.1, -122.5]
        north_east = [47.8, -121.0]
        
        m = folium.Map(location=[47.5112, -122.257], zoom_start=10, min_zoom=9, max_bounds=True)
        m.fit_bounds([south_west, north_east])
        
        # Tweak 3: Draw the King County Boundary on the map
        boundary_data = get_king_county_geojson()
        if boundary_data:
            folium.GeoJson(
                boundary_data,
                style_function=lambda x: {'fillColor': '#3186cc', 'color': '#0000FF', 'weight': 2, 'fillOpacity': 0.1}
            ).add_to(m)
            
        m.add_child(folium.LatLngPopup())
        map_data = st_folium(m, height=350, use_container_width=True)
        
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            long = map_data["last_clicked"]["lng"]
        else:
            lat = 47.5112
            long = -122.257
        st.caption(f"**Coordinates Selected:** Latitude {lat:.4f}, Longitude {long:.4f}")
        
    else:
        st.info("Input exact coordinates. King County generally spans Lat 47.1 to 47.8, and Long -122.5 to -121.0.")
        man_col1, man_col2 = st.columns(2)
        with man_col1:
            lat = st.number_input("Latitude", value=47.5112, format="%.4f")
        with man_col2:
            long = st.number_input("Longitude", value=-122.257, format="%.4f")

with top_col2:
    st.subheader("📊 Valuation")
    with st.container(border=True):
        if st.session_state.prediction_data is None:
            st.info("👈 Select a location and fill out the details below to generate a valuation.")
            st.write("\n" * 6)
        else:
            data = st.session_state.prediction_data
            price = data["predicted_price"]
            model_used = data["model_used"]
            rmse = data["rmse"]
            
            st.metric(label="Estimated Market Value", value=f"${price:,.0f}")
            
            show_interval = st.toggle("Show Confidence Interval")
            if show_interval:
                lower_bound = max(0, price - rmse)
                upper_bound = price + rmse
                st.markdown(f"**Expected Range:**\n\n `${lower_bound:,.0f}` — `${upper_bound:,.0f}`")
            
            st.divider()
            st.caption(f"⚙️ **Engine:** {model_used}")
            if show_interval:
                st.caption(f"📉 **Model Error Margin (RMSE):** ±${rmse:,.0f}")

st.write("")

# --- BOTTOM ROW: PROPERTY DETAILS ---
st.subheader("🏠 Property Details")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("**Interior Basics**")
        bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.25)
        floors = st.number_input("Floors", min_value=1.0, value=1.0, step=0.5)

with col2:
    with st.container(border=True):
        st.markdown("**Size & Area**")
        sqft_living = st.number_input("Living Area (Sqft)", min_value=1, value=2000, step=100)
        sqft_lot = st.number_input("Lot Size (Sqft)", min_value=1, value=5000, step=100)
        sqft_basement = st.number_input("Basement (Sqft)", min_value=0, value=0, step=100)
        sqft_above = sqft_living - sqft_basement

with col3:
    with st.container(border=True):
        st.markdown("**Condition & History**")
        grade = st.slider("Construction Grade", min_value=1, max_value=13, value=7)
        condition = st.slider("Condition", min_value=1, max_value=5, value=3)
        yr_built = st.number_input("Year Built", min_value=1900, max_value=CURRENT_YEAR, value=1990)

with st.expander("Advanced Features (Views & Neighbors)"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        waterfront = st.selectbox("Waterfront", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        view = st.slider("View Score", 0, 4, 0)
        zipcode = st.number_input("Zipcode", min_value=98000, max_value=99000, value=98103, step=1)
    with adv_col2:
        yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=CURRENT_YEAR, value=0, step=1)
        sqft_living15 = st.number_input("Avg Neighbor Living Area", min_value=1, value=1800, step=100)
        sqft_lot15 = st.number_input("Avg Neighbor Lot Size", min_value=1, value=5000, step=100)

# --- THE BIG BUTTON ---
st.write("")
if st.button("Generate Valuation", type="primary", use_container_width=True):
    payload = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
        "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront,
        "view": view, "condition": condition, "grade": grade,
        "sqft_above": sqft_above, "sqft_basement": sqft_basement, "yr_built": yr_built,
        "yr_renovated": yr_renovated, "zipcode": zipcode, "lat": lat,
        "long": long, "sqft_living15": sqft_living15, "sqft_lot15": sqft_lot15
    }
    
    with st.spinner("Analyzing market data..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            st.session_state.prediction_data = response.json()
            
            # Tweak 2: Trigger the toast and instantly scroll to the top
            st.session_state.show_success_toast = True
            st.rerun() 
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the FastAPI backend: {e}")
            st.session_state.prediction_data = None