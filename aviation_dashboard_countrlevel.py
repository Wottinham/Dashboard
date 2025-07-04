import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Model configuration
# ----------------------

EMISSION_FACTOR = 0.115  # default kg CO₂ per pax-km
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND = 1.4

# ----------------------
# Helper – generate dummy data when no CSV is provided
# ----------------------
def load_dummy_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    origins = ["Germany", "France", "United States", "Japan"]
    destinations = ["Spain", "Italy", "United Kingdom", "Canada"]
    rows = []
    for o in origins:
        for d in destinations:
            if o == d:
                continue
            rows.append(
                {
                    "Origin Country Name": o,
                    "Destination Country Name": d,
                    "Origin Airport": f"{o[:3].upper()}-INTL",
                    "Destination Airport": f"{d[:3].upper()}-INTL",
                    "Distance (km)": int(rng.integers(500, 9000)),
                    "Passengers": int(rng.integers(50000, 1000000)),
                    "Avg. Total Fare(USD)": round(rng.uniform(150, 700), 2),
                }
            )
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between airports.")

uploaded_file = st.sidebar.file_uploader(
    "Upload airport-pair passenger CSV", type=["csv"]
)

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("📈 Policy & Economic Inputs")

# Emission factor slider
emission_factor = st.sidebar.slider(
    "Emission factor (kg CO₂ per pax‑km)",
    0.0, 1.0, EMISSION_FACTOR, 0.001,
    help="Emissions per passenger‑kilometer."
)

# Flat air passenger tax
air_passenger_tax = st.sidebar.slider(
    "Air passenger tax (USD)",
    0, 100, 0, 1,
    help="Flat tax added to each ticket."
)

# Carbon price slider
ets_price = st.sidebar.slider(
    "Carbon price (EUR / tCO₂)",
    0, 400, 100, 5,
    help="Select the carbon price applied to aviation emissions.",
)

# ----------------------
# CO₂ Pricing Countries (Origin & Destination)
# ----------------------
st.sidebar.markdown("### Apply CO₂ Price to Selected Countries")

origin_all = sorted(df["Origin Country Name"].unique())
# Note: df not yet defined; move data load above or define origins later
