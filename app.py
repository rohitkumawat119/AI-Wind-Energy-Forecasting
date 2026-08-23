
import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Wind Energy Forecasting",
    page_icon="🌬️",
    layout="wide"
)


# ============================================================
# LOAD PRODUCTION MODEL AND FILES
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "wind_gru_production.keras"
    )

    feature_scaler = joblib.load(
        "wind_feature_scaler.pkl"
    )

    target_scaler = joblib.load(
        "wind_target_scaler.pkl"
    )

    with open("wind_model_config.json", "r") as f:
        config = json.load(f)

    return model, feature_scaler, target_scaler, config


model, feature_scaler, target_scaler, config = load_model()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

AIR_DENSITY = config["air_density"]
ROTOR_DIAMETER = config["rotor_diameter"]
POWER_COEFFICIENT = config["power_coefficient"]
SEQ_LEN = config["sequence_length"]

# These are the exact feature names used during model training
MODEL_FEATURES = config["features"]


# ============================================================
# WIND POWER CALCULATION
# ============================================================

ROTOR_AREA = np.pi * (ROTOR_DIAMETER / 2) ** 2


def calculate_wind_power(wind_speed_kmh):

    wind_speed_ms = wind_speed_kmh / 3.6

    power_watts = (
        0.5
        * AIR_DENSITY
        * ROTOR_AREA
        * (wind_speed_ms ** 3)
        * POWER_COEFFICIENT
    )

    return power_watts / 1000


# ============================================================
# WIND POTENTIAL CLASSIFICATION
# ============================================================

def classify_wind_potential(wind_speed_kmh):

    if wind_speed_kmh < 10:
        return "Low"

    elif wind_speed_kmh < 20:
        return "Moderate"

    elif wind_speed_kmh < 30:
        return "High"

    else:
        return "Very High"


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🌬️ AI-Based Wind Energy Forecasting Platform")

st.write(
    "Predict wind speed and estimate theoretical wind power "
    "using a GRU deep-learning model."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Location")

city = st.sidebar.text_input(
    "Enter City",
    "Jaipur"
)

predict_button = st.sidebar.button(
    "Predict Wind Energy"
)


# ============================================================
# MAIN PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # STEP 1: GEOCODING
    # ========================================================

    geocode_url = (
        "https://nominatim.openstreetmap.org/search"
    )

    geocode_params = {
        "q": city,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "AI-Wind-Energy-Forecasting-App"
    }

    try:

        geo_response = requests.get(
            geocode_url,
            params=geocode_params,
            headers=headers,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if not geo_data:

            st.error(
                f"Could not find location: {city}"
            )

            st.stop()

        latitude = float(
            geo_data[0]["lat"]
        )

        longitude = float(
            geo_data[0]["lon"]
        )

    except Exception as e:

        st.error(
            f"Location lookup failed: {e}"
        )

        st.stop()


    # ========================================================
    # STEP 2: GET WEATHER DATA FROM OPEN-METEO
    # ========================================================

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    # IMPORTANT:
    # Open-Meteo API uses variable names WITHOUT units.
    #
    # These are converted to the exact training column names
    # after the API response is received.

    api_features = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "pressure_msl",
        "surface_pressure",
        "vapour_pressure_deficit",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_direction_100m",
        "wind_gusts_10m",
        "wind_speed_100m"
    ]

    weather_params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": ",".join(api_features),

        "forecast_days": 1,

        "timezone": "auto"
    }


    try:

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=15
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

    except Exception as e:

        st.error(
            f"Weather API request failed: {e}"
        )

        st.stop()


    # ========================================================
    # STEP 3: PREPARE WEATHER DATA
    # ========================================================

    try:

        weather_df = pd.DataFrame(
            weather_data["hourly"]
        )

        weather_df["time"] = pd.to_datetime(
            weather_df["time"]
        )

        weather_df = weather_df.sort_values(
            "time"
        ).reset_index(drop=True)

    except Exception as e:

        st.error(
            f"Could not prepare weather data: {e}"
        )

        st.stop()


    # ========================================================
    # STEP 4: RENAME API COLUMNS TO TRAINING COLUMN NAMES
    # ========================================================

    column_mapping = {

        "temperature_2m":
            "temperature_2m (°C)",

        "relative_humidity_2m":
            "relative_humidity_2m (%)",

        "dew_point_2m":
            "dew_point_2m (°C)",

        "pressure_msl":
            "pressure_msl (hPa)",

        "surface_pressure":
            "surface_pressure (hPa)",

        "vapour_pressure_deficit":
            "vapour_pressure_deficit (kPa)",

        "wind_speed_10m":
            "wind_speed_10m (km/h)",

        "wind_direction_10m":
            "wind_direction_10m (°)",

        "wind_direction_100m":
            "wind_direction_100m (°)",

        "wind_gusts_10m":
            "wind_gusts_10m (km/h)",

        "wind_speed_100m":
            "wind_speed_100m (km/h)"
    }

    weather_df = weather_df.rename(
        columns=column_mapping
    )


    # ========================================================
    # STEP 5: CHECK REQUIRED MODEL FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in weather_df.columns
    ]

    if missing_features:

        st.error(
            "The following model features are missing "
            "from the weather data:"
        )

        st.write(missing_features)

        st.stop()


    # ========================================================
    # STEP 6: CHECK 24-HOUR SEQUENCE
    # ========================================================

    if len(weather_df) < SEQ_LEN:

        st.error(
            f"Not enough hourly weather data. "
            f"The GRU requires {SEQ_LEN} hours."
        )

        st.stop()


    # ========================================================
    # STEP 7: CREATE MODEL INPUT
    # ========================================================

    latest_weather = weather_df[
        MODEL_FEATURES
    ].tail(
        SEQ_LEN
    )


    # Scale using the SAME scaler used during training
    X_live = feature_scaler.transform(
        latest_weather.values
    )


    # Add batch dimension
    #
    # Before:
    # (24, 10)
    #
    # After:
    # (1, 24, 10)

    X_live = np.expand_dims(
        X_live,
        axis=0
    )


    # ========================================================
    # STEP 8: GRU PREDICTION
    # ========================================================

    prediction_scaled = model.predict(
        X_live,
        verbose=0
    )


    # Convert scaled prediction back to km/h

    predicted_wind_speed = (
        target_scaler
        .inverse_transform(prediction_scaled)[0][0]
    )


    # Make sure prediction isn't negative

    predicted_wind_speed = max(
        0,
        float(predicted_wind_speed)
    )


    # ========================================================
    # STEP 9: CALCULATE WIND POWER
    # ========================================================

    predicted_power = calculate_wind_power(
        predicted_wind_speed
    )


    # ========================================================
    # STEP 10: CLASSIFY WIND POTENTIAL
    # ========================================================

    wind_potential = classify_wind_potential(
        predicted_wind_speed
    )


    # ========================================================
    # DISPLAY LOCATION
    # ========================================================

    st.subheader(
        f"📍 {city.title()}"
    )

    st.caption(
        f"Latitude: {latitude:.4f} | "
        f"Longitude: {longitude:.4f}"
    )


    # ========================================================
    # METRIC CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted Wind Speed",
            f"{predicted_wind_speed:.2f} km/h"
        )


    with col2:

        st.metric(
            "Estimated Wind Power",
            f"{predicted_power:.2f} kW"
        )


    with col3:

        st.metric(
            "Wind Potential",
            wind_potential
        )


    # ========================================================
    # POWER ASSUMPTION INFORMATION
    # ========================================================

    st.info(
        "Estimated power is theoretical and is calculated "
        f"using a {ROTOR_DIAMETER:.0f} m rotor diameter, "
        f"Cp = {POWER_COEFFICIENT:.2f}, and air density "
        f"of {AIR_DENSITY:.3f} kg/m³."
    )


    # ========================================================
    # 24-HOUR WIND SPEED CHART
    # ========================================================

    st.subheader(
        "24-Hour Wind Speed"
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=weather_df["time"],
            y=weather_df["wind_speed_100m (km/h)"],
            mode="lines+markers",
            name="Wind Speed at 100m"
        )
    )


    # Add predicted value as a horizontal reference
    # so the user can compare the GRU prediction
    # with the available 24-hour wind profile.

    fig.add_hline(
        y=predicted_wind_speed,
        line_dash="dash",
        annotation_text=(
            f"GRU Prediction: "
            f"{predicted_wind_speed:.2f} km/h"
        )
    )


    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Wind Speed (km/h)",
        hovermode="x unified",
        height=450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # CURRENT WEATHER INFORMATION
    # ========================================================

    st.subheader(
        "Current Weather Data"
    )


    latest = weather_df.iloc[-1]


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Temperature",
            f"{latest['temperature_2m (°C)']:.1f} °C"
        )


    with col2:

        st.metric(
            "Humidity",
            f"{latest['relative_humidity_2m (%)']:.0f}%"
        )


    with col3:

        st.metric(
            "Wind at 10m",
            f"{latest['wind_speed_10m (km/h)']:.1f} km/h"
        )


    with col4:

        st.metric(
            "Wind at 100m",
            f"{latest['wind_speed_100m (km/h)']:.1f} km/h"
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "Model Information"
    )


    info_col1, info_col2, info_col3 = st.columns(3)


    with info_col1:

        st.metric(
            "Production Model",
            "GRU"
        )


    with info_col2:

        st.metric(
            "Lookback Window",
            f"{SEQ_LEN} hours"
        )


    with info_col3:

        st.metric(
            "Input Features",
            len(MODEL_FEATURES)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Wind Energy Forecasting Platform | "
    "Production Model: GRU | "
    "Lookback: 24 hours"
)
