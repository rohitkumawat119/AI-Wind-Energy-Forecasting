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
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --navy: #071426;
        --navy-2: #0b1d32;
        --panel: #10253b;
        --panel-light: #142d45;
        --teal: #35d0ba;
        --teal-soft: #a0f3e5;
        --text: #eef7fb;
        --muted: #9bb0c3;
        --border: rgba(155, 190, 210, 0.17);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 0%, rgba(23, 125, 139, 0.16), transparent 28rem),
            linear-gradient(145deg, #071426 0%, #091a2d 48%, #071426 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 20, 38, 0.78);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1d32 0%, #081626 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.025em;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .hero {
        padding: 1.55rem 1.7rem;
        border: 1px solid rgba(53, 208, 186, 0.25);
        border-radius: 22px;
        background:
            radial-gradient(circle at 88% 25%, rgba(53, 208, 186, 0.15), transparent 18rem),
            linear-gradient(120deg, rgba(16, 37, 59, 0.97), rgba(8, 26, 43, 0.96));
        margin: 0.35rem 0 1.45rem 0;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.15);
    }

    .eyebrow {
        color: var(--teal) !important;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.55rem;
    }

    .hero-title {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        line-height: 1.13;
        font-weight: 700;
        margin: 0 0 0.65rem 0;
    }

    .hero-copy {
        color: #b2c5d5;
        max-width: 700px;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }

    .section-kicker {
        color: var(--teal);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 0.25rem;
    }

    .section-title {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        margin: 0 0 0.75rem 0;
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(18, 43, 66, 0.98), rgba(12, 31, 50, 0.98));
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 1.1rem 1.15rem;
        min-height: 142px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.12);
    }

    .metric-label {
        color: #a9bdce;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }

    .metric-value {
        color: #f4fbff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.55rem, 2.2vw, 2rem);
        font-weight: 700;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: var(--teal-soft);
        font-size: 0.76rem;
        margin-top: 0.55rem;
    }

    .info-panel {
        background: rgba(16, 37, 59, 0.75);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        height: 100%;
    }

    .info-label {
        color: #9bb0c3;
        font-size: 0.77rem;
        margin-bottom: 0.25rem;
    }

    .info-value {
        color: #edf7fc;
        font-size: 1.05rem;
        font-weight: 600;
    }

    .pill {
        display: inline-block;
        padding: 0.32rem 0.7rem;
        border: 1px solid rgba(53, 208, 186, 0.4);
        border-radius: 999px;
        background: rgba(53, 208, 186, 0.09);
        color: #8debdc;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .empty-state {
        border: 1px dashed rgba(155, 190, 210, 0.28);
        border-radius: 18px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        background: rgba(16, 37, 59, 0.38);
    }

    .empty-icon {
        font-size: 2.2rem;
        margin-bottom: 0.6rem;
    }

    .empty-title {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .empty-copy {
        color: var(--muted);
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.55;
    }

    div.stButton > button {
        width: 100%;
        border: 1px solid rgba(53, 208, 186, 0.6);
        border-radius: 10px;
        background: linear-gradient(110deg, #35d0ba, #19aaad);
        color: #041a27;
        font-weight: 800;
        padding: 0.68rem 1rem;
        min-height: 2.8rem;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #8df1e3;
        background: linear-gradient(110deg, #73e7d7, #35d0ba);
        color: #041a27;
        transform: translateY(-1px);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #10253b;
        border-color: rgba(155, 190, 210, 0.25);
        border-radius: 9px;
    }

    div[data-testid="stMetric"] {
        background: rgba(16, 37, 59, 0.65);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bdce;
    }

    div[data-testid="stMetricValue"] {
        color: #f4fbff;
    }

    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
        background: rgba(16, 37, 59, 0.65);
        padding: 0.35rem;
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        text-align: center;
        color: #7890a5;
        font-size: 0.78rem;
        padding: 1.1rem 0 0.5rem 0;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 1.2rem;
        }
        .metric-card {
            min-height: 125px;
            padding: 0.9rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD PRODUCTION MODEL AND FILES
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("wind_gru_production.keras")
    feature_scaler = joblib.load("wind_feature_scaler.pkl")
    target_scaler = joblib.load("wind_target_scaler.pkl")

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

# These are the exact feature names used during model training.
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
# REUSABLE UI HELPERS
# ============================================================

def render_metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(kicker, title):
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:2rem; margin-bottom:0.25rem;">🌬️</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.22rem;
                    font-weight:700; color:#eef7fb;">Wind Intelligence</div>
        <div style="color:#9bb0c3; font-size:0.83rem; margin:0.25rem 0 1.4rem 0;">
            AI forecasting workspace
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Forecast location")
    city = st.text_input(
        "City or location",
        value="Jaipur",
        placeholder="e.g. Jaipur, India",
        help="Enter a city or place that can be found by OpenStreetMap.",
    )

    predict_button = st.button("🌬️  Predict wind energy", use_container_width=True)

    st.markdown("---")
    st.markdown("**Model at a glance**")
    st.markdown(
        f"""
        <div class="info-panel">
            <div class="info-label">Production model</div>
            <div class="info-value">GRU deep learning</div>
            <div style="height:0.75rem"></div>
            <div class="info-label">Lookback window</div>
            <div class="info-value">{SEQ_LEN} hourly steps</div>
            <div style="height:0.75rem"></div>
            <div class="info-label">Input features</div>
            <div class="info-value">{len(MODEL_FEATURES)} weather variables</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Weather data: Open-Meteo · Location search: OpenStreetMap")


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Renewable energy intelligence</div>
        <div class="hero-title">AI Wind Energy<br>Forecasting Platform</div>
        <p class="hero-copy">
            Explore local wind conditions, forecast wind speed with a production
            GRU model, and estimate theoretical wind power from the predicted speed.
        </p>
        <div style="margin-top:1rem;">
            <span class="pill">● GRU model</span>
            &nbsp; <span class="pill">Live weather inputs</span>
            &nbsp; <span class="pill">Hourly analytics</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN PREDICTION
# ============================================================

if not predict_button:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🧭</div>
            <div class="empty-title">Ready to explore wind potential?</div>
            <div class="empty-copy">
                Enter a city in the sidebar and select <b>Predict wind energy</b>.
                The dashboard will retrieve hourly weather data, run the GRU forecast,
                and display estimated wind power and weather analytics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    render_section_heading("How it works", "From weather data to wind insight")
    flow1, flow2, flow3 = st.columns(3)

    with flow1:
        st.markdown(
            """
            <div class="info-panel">
                <div class="section-kicker">01 · Retrieve</div>
                <div class="info-value">Local weather</div>
                <p>Find the location and retrieve hourly weather variables from Open-Meteo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with flow2:
        st.markdown(
            f"""
            <div class="info-panel">
                <div class="section-kicker">02 · Forecast</div>
                <div class="info-value">GRU prediction</div>
                <p>Scale the latest {SEQ_LEN}-hour feature sequence using the training scaler and run the GRU model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with flow3:
        st.markdown(
            """
            <div class="info-panel">
                <div class="section-kicker">03 · Estimate</div>
                <div class="info-value">Wind power</div>
                <p>Convert predicted wind speed into a theoretical power estimate using the configured turbine assumptions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if predict_button:
    if not city.strip():
        st.error("Please enter a city or location in the sidebar.")
        st.stop()

    # ========================================================
    # STEP 1: GEOCODING
    # ========================================================

    geocode_url = "https://nominatim.openstreetmap.org/search"
    geocode_params = {
        "q": city.strip(),
        "format": "json",
        "limit": 1,
    }
    headers = {
        "User-Agent": "AI-Wind-Energy-Forecasting-App"
    }

    with st.spinner(f"Finding {city.strip()}..."):
        try:
            geo_response = requests.get(
                geocode_url,
                params=geocode_params,
                headers=headers,
                timeout=10,
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            if not geo_data:
                st.error(f"Could not find location: {city}")
                st.stop()

            latitude = float(geo_data[0]["lat"])
            longitude = float(geo_data[0]["lon"])

        except requests.RequestException as e:
            st.error(f"Location lookup failed. Please try again. Details: {e}")
            st.stop()
        except (ValueError, KeyError, TypeError, IndexError) as e:
            st.error(f"Could not read the location result. Details: {e}")
            st.stop()

    # ========================================================
    # STEP 2: GET WEATHER DATA FROM OPEN-METEO
    # ========================================================

    weather_url = "https://api.open-meteo.com/v1/forecast"

    # Open-Meteo API variable names do not include units.
    # They are renamed to the exact training column names below.
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
        "wind_speed_100m",
    ]

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(api_features),
        "forecast_days": 1,
        "timezone": "auto",
    }

    with st.spinner("Retrieving hourly weather and preparing the forecast..."):
        try:
            weather_response = requests.get(
                weather_url,
                params=weather_params,
                timeout=15,
            )
            weather_response.raise_for_status()
            weather_data = weather_response.json()
        except requests.RequestException as e:
            st.error(f"Weather API request failed. Please try again. Details: {e}")
            st.stop()

    # ========================================================
    # STEP 3: PREPARE WEATHER DATA
    # ========================================================

    try:
        weather_df = pd.DataFrame(weather_data["hourly"])
        weather_df["time"] = pd.to_datetime(weather_df["time"])
        weather_df = weather_df.sort_values("time").reset_index(drop=True)
    except (KeyError, TypeError, ValueError) as e:
        st.error(f"Could not prepare weather data: {e}")
        st.stop()

    # ========================================================
    # STEP 4: RENAME API COLUMNS TO TRAINING COLUMN NAMES
    # ========================================================

    column_mapping = {
        "temperature_2m": "temperature_2m (°C)",
        "relative_humidity_2m": "relative_humidity_2m (%)",
        "dew_point_2m": "dew_point_2m (°C)",
        "pressure_msl": "pressure_msl (hPa)",
        "surface_pressure": "surface_pressure (hPa)",
        "vapour_pressure_deficit": "vapour_pressure_deficit (kPa)",
        "wind_speed_10m": "wind_speed_10m (km/h)",
        "wind_direction_10m": "wind_direction_10m (°)",
        "wind_direction_100m": "wind_direction_100m (°)",
        "wind_gusts_10m": "wind_gusts_10m (km/h)",
        "wind_speed_100m": "wind_speed_100m (km/h)",
    }

    weather_df = weather_df.rename(columns=column_mapping)

    # ========================================================
    # STEP 5: CHECK REQUIRED MODEL FEATURES
    # ========================================================

    missing_features = [
        feature for feature in MODEL_FEATURES
        if feature not in weather_df.columns
    ]

    if missing_features:
        st.error("Some required model features are missing from the weather data.")
        st.write(missing_features)
        st.stop()

    # ========================================================
    # STEP 6: CHECK SEQUENCE LENGTH
    # ========================================================

    if len(weather_df) < SEQ_LEN:
        st.error(
            f"Not enough hourly weather data. The GRU requires {SEQ_LEN} hours, "
            f"but the API returned {len(weather_df)}."
        )
        st.stop()

    # ========================================================
    # STEP 7: CREATE MODEL INPUT
    # ========================================================

    latest_weather = weather_df[MODEL_FEATURES].tail(SEQ_LEN)

    # Use the same feature scaler used during training.
    X_live = feature_scaler.transform(latest_weather.values)

    # Add batch dimension: (sequence length, features) -> (1, sequence length, features).
    X_live = np.expand_dims(X_live, axis=0)

    # ========================================================
    # STEP 8: GRU PREDICTION
    # ========================================================

    prediction_scaled = model.predict(X_live, verbose=0)

    # Convert the scaled prediction back to km/h.
    predicted_wind_speed = (
        target_scaler.inverse_transform(prediction_scaled)[0][0]
    )

    # Make sure the prediction is not negative.
    predicted_wind_speed = max(0, float(predicted_wind_speed))

    # ========================================================
    # STEP 9: CALCULATE WIND POWER
    # ========================================================

    predicted_power = calculate_wind_power(predicted_wind_speed)

    # ========================================================
    # STEP 10: CLASSIFY WIND POTENTIAL
    # ========================================================

    wind_potential = classify_wind_potential(predicted_wind_speed)

    # ========================================================
    # LOCATION SUMMARY
    # ========================================================

    st.markdown("")
    render_section_heading("Forecast location", city.strip().title())
    st.caption(f"Coordinates: {latitude:.4f}°, {longitude:.4f}°")

    # ========================================================
    # PRIMARY METRIC CARDS
    # ========================================================

    render_section_heading("Prediction overview", "Your wind energy snapshot")
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        render_metric_card(
            "PREDICTED WIND SPEED",
            f"{predicted_wind_speed:.2f} <span style='font-size:0.9rem;font-weight:500'>km/h</span>",
            "GRU model output",
        )

    with col2:
        render_metric_card(
            "ESTIMATED WIND POWER",
            f"{predicted_power:.2f} <span style='font-size:0.9rem;font-weight:500'>kW</span>",
            "Theoretical power estimate",
        )

    with col3:
        render_metric_card(
            "WIND POTENTIAL",
            wind_potential,
            "Based on predicted wind speed",
        )

    st.markdown("")

    # ========================================================
    # POWER ASSUMPTIONS
    # ========================================================

    with st.expander("About the estimated wind power", expanded=False):
        st.write(
            "This is a theoretical estimate calculated from the predicted wind speed. "
            "It is not a measured turbine output or a guarantee of electricity generation."
        )
        assumption1, assumption2, assumption3 = st.columns(3)
        with assumption1:
            st.metric("Rotor diameter", f"{ROTOR_DIAMETER:.0f} m")
        with assumption2:
            st.metric("Power coefficient (Cp)", f"{POWER_COEFFICIENT:.2f}")
        with assumption3:
            st.metric("Air density", f"{AIR_DENSITY:.3f} kg/m³")
        st.caption(
            "The calculation does not account for turbine-specific cut-in/cut-out speeds, "
            "electrical losses, or changing operating efficiency."
        )

    # ========================================================
    # WIND SPEED CHART
    # ========================================================

    st.markdown("")
    render_section_heading("Hourly analytics", "Wind speed profile")

    chart_df = weather_df.copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["time"],
            y=chart_df["wind_speed_100m (km/h)"],
            mode="lines+markers",
            name="Wind at 100 m",
            line=dict(color="#35d0ba", width=3, shape="spline"),
            marker=dict(size=5, color="#a0f3e5", line=dict(width=1, color="#10253b")),
            fill="tozeroy",
            fillcolor="rgba(53, 208, 186, 0.10)",
            hovertemplate="%{x|%a, %H:%M}<br><b>%{y:.1f} km/h</b><extra></extra>",
        )
    )

    fig.add_hline(
        y=predicted_wind_speed,
        line_dash="dash",
        line_color="#f2b866",
        line_width=2,
        annotation_text=f"GRU prediction · {predicted_wind_speed:.2f} km/h",
        annotation_position="top left",
        annotation_font_color="#f2c982",
    )

    fig.update_layout(
        height=410,
        margin=dict(l=15, r=20, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#b5c8d8", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#dce9f2"),
        ),
        xaxis=dict(
            title="Local time",
            showgrid=False,
            linecolor="rgba(155,190,210,0.18)",
            tickfont=dict(color="#9bb0c3"),
            title_font=dict(color="#9bb0c3"),
        ),
        yaxis=dict(
            title="Wind speed (km/h)",
            gridcolor="rgba(155,190,210,0.12)",
            zeroline=False,
            tickfont=dict(color="#9bb0c3"),
            title_font=dict(color="#9bb0c3"),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ========================================================
    # CURRENT WEATHER
    # ========================================================

    st.markdown("")
    render_section_heading("Live conditions", "Latest weather snapshot")

    latest = weather_df.iloc[-1]
    weather1, weather2, weather3, weather4 = st.columns(4, gap="medium")

    with weather1:
        render_metric_card(
            "TEMPERATURE",
            f"{latest['temperature_2m (°C)']:.1f} °C",
            "At 2 m above ground",
        )

    with weather2:
        render_metric_card(
            "RELATIVE HUMIDITY",
            f"{latest['relative_humidity_2m (%)']:.0f}%",
            "Near-surface humidity",
        )

    with weather3:
        render_metric_card(
            "WIND SPEED · 10 M",
            f"{latest['wind_speed_10m (km/h)']:.1f} km/h",
            "Near-surface wind",
        )

    with weather4:
        render_metric_card(
            "WIND SPEED · 100 M",
            f"{latest['wind_speed_100m (km/h)']:.1f} km/h",
            "Higher-altitude wind",
        )

    # ========================================================
    # MODEL DETAILS
    # ========================================================

    st.markdown("")
    with st.expander("Model and input details"):
        detail1, detail2, detail3 = st.columns(3)
        with detail1:
            st.metric("Production model", "GRU")
        with detail2:
            st.metric("Lookback window", f"{SEQ_LEN} hours")
        with detail3:
            st.metric("Input features", len(MODEL_FEATURES))

        st.markdown("**Features used by the model**")
        st.write(", ".join(MODEL_FEATURES))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        AI Wind Energy Forecasting Platform &nbsp;·&nbsp; GRU deep-learning model
        <br>
        Weather inputs from Open-Meteo · Theoretical power estimates are not measured output.
    </div>
    """,
    unsafe_allow_html=True,
)
