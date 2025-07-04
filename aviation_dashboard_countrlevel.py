import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from keplergl import KeplerGl
import streamlit.components.v1 as components
import json

# ----------------------
# Defaults
# ----------------------
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND = 1.4

# ----------------------
# Load Dummy Data
# ----------------------
def load_dummy_data():
    rng = np.random.default_rng(seed=42)
    origins = ["Germany", "France", "United States", "Japan"]
    destinations = ["Spain", "Italy", "United Kingdom", "Canada"]
    rows = []
    for o in origins:
        for d in destinations:
            if o == d:
                continue
            rows.append({
                "Origin Country Name": o,
                "Destination Country Name": d,
                "Origin Airport": f"{o[:3].upper()}-INTL",
                "Destination Airport": f"{d[:3].upper()}-INTL",
                "Distance (km)": int(rng.integers(500, 9000)),
                "Passengers": int(rng.integers(50000, 1000000)),
                "Avg. Total Fare(USD)": round(rng.uniform(150, 700), 2),
            })
    return pd.DataFrame(rows)

# ----------------------
# Streamlit Setup
# ----------------------
st.set_page_config(page_title="JETPAS - Aviation Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between airport pairs with taxes and carbon pricing.")

# ----------------------
# Upload CSV Data
# ----------------------
st.sidebar.header("📄 Upload Airport Pair CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded.")
else:
    df = load_dummy_data()
    st.info("ℹ️ Using sample dummy data.")

# ----------------------
# Upload Coordinate File
# ----------------------
st.sidebar.markdown("📍 Optional: Upload Airport Coordinates Excel")
coord_file = st.sidebar.file_uploader("Upload Coordinates (.xlsx)", type=["xlsx"])

coord_df = pd.DataFrame()
if coord_file:
    try:
        coord_df = pd.read_excel(coord_file, engine='openpyxl')
        st.success("📌 Coordinates loaded.")
    except Exception as e:
        st.error(f"❌ Failed to process coordinate file: {e}")

# ----------------------
# Validate Columns
# ----------------------
required_cols = {
    "Origin Country Name", "Destination Country Name",
    "Origin Airport", "Destination Airport", "Distance (km)",
    "Passengers", "Avg. Total Fare(USD)"
}
if not required_cols.issubset(df.columns):
    st.error("❌ CSV is missing required columns.")
    st.stop()

df = df.dropna(subset=required_cols)

# ----------------------
# Origin & Destination Lists
# ----------------------
origin_all = sorted(df["Origin Country Name"].unique())
dest_all = sorted(df["Destination Country Name"].unique())

# ----------------------
# Carbon Pricing Sidebar
# ----------------------
st.sidebar.markdown("### 💨 Carbon Pricing")
carbon_policy = st.sidebar.selectbox("Enable Carbon Pricing?", ["Disable", "Enable"])
if carbon_policy == "Enable":
    ets_price = st.sidebar.slider("Carbon price (EUR / tCO₂)", 0, 400, 100, 5)
    carbon_origin_countries = st.sidebar.multiselect("Taxed Origin Countries", origin_all, default=origin_all)
    carbon_dest_countries = st.sidebar.multiselect("Taxed Destination Countries", dest_all, default=dest_all)
else:
    ets_price = 0.0
    carbon_origin_countries = []
    carbon_dest_countries = []

# ----------------------
# Air Passenger Tax Sidebar
# ----------------------
st.sidebar.markdown("### ✈️ Air Passenger Tax")
tax_policy = st.sidebar.selectbox("Enable Tax?", ["Disable", "Enable"])
if tax_policy == "Enable":
    air_tax = st.sidebar.slider("Air Passenger Tax (USD)", 0, 100, 0, 1)
    tax_origin_countries = st.sidebar.multiselect("Taxed Origin Countries", origin_all, default=origin_all)
    tax_dest_countries = st.sidebar.multiselect("Taxed Destination Countries", dest_all, default=dest_all)
else:
    air_tax = 0.0
    tax_origin_countries = []
    tax_dest_countries = []

# ----------------------
# Parameters
# ----------------------
st.sidebar.markdown("### ⚙️ Parameters")
pass_through = st.sidebar.slider("Cost Pass-Through to Fares (%)", 0, 100, 80, 5) / 100
emission_factor = st.sidebar.slider("Emission Factor (kg CO₂ / pax-km)", 0.0, 1.0, 0.115, 0.001)

# ----------------------
# Economics
# ----------------------
global_gdp_growth = st.sidebar.slider("Global GDP Growth (%)", -5.0, 8.0, 2.5, 0.1)
user_price_elasticity = st.sidebar.slider("Price Elasticity", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1)
user_gdp_elasticity = st.sidebar.slider("GDP Elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1)

st.sidebar.markdown("### 📊 Per-Country GDP Adjustment")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth"):
    for country in origin_all:
        gdp_growth_by_country[country] = st.slider(
            f"{country}", -5.0, 8.0, global_gdp_growth, 0.1, key=f"gdp_{country}"
        )

# ----------------------
# Policy Effects
# ----------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor

df["Carbon cost per pax"] = 0.0
df["Air passenger tax per pax"] = 0.0

# Carbon cost
if carbon_policy == "Enable":
    mask = df["Origin Country Name"].isin(carbon_origin_countries) & df["Destination Country Name"].isin(carbon_dest_countries)
    df.loc[mask, "Carbon cost per pax"] = df.loc[mask, "CO2 per pax (kg)"] / 1000 * ets_price * pass_through

# Tax cost
if tax_policy == "Enable":
    mask = df["Origin Country Name"].isin(tax_origin_countries) & df["Destination Country Name"].isin(tax_dest_countries)
    df.loc[mask, "Air passenger tax per pax"] = air_tax * pass_through

# New fare
df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"] + df["Air passenger tax per pax"]
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

# Elasticity Model
fare_factor = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"]).replace([np.inf, -np.inf], np.nan) ** user_price_elasticity
df["GDP Growth (%)"] = df["Origin Country Name"].map(gdp_growth_by_country).fillna(global_gdp_growth)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"] / 100) ** user_gdp_elasticity
df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# ----------------------
# Output Table
# ----------------------
st.subheader("📊 Results Table")
st.dataframe(
    df[[
        "Origin Airport", "Destination Airport",
        "Passengers", "Passengers after policy", "Passenger Δ (%)",
        "Avg. Total Fare(USD)", "New Avg Fare", "Fare Δ (%)",
        "Carbon cost per pax", "Air passenger tax per pax"
    ]],
    use_container_width=True
)

# ----------------------
# Kepler Map
# ----------------------
if not coord_df.empty and {'IATA_Code', 'DecLat', 'DecLon'}.issubset(coord_df.columns):
    # Merge coordinates
    coords = coord_df.rename(columns={'IATA_Code': 'Airport'})
    df_map = df.copy()
    df_map["Origin IATA"] = df_map["Origin Airport"].str[:3]
    df_map["Destination IATA"] = df_map["Destination Airport"].str[:3]

    df_map = df_map.merge(coords, left_on="Origin IATA", right_on="Airport", how="left")
    df_map = df_map.rename(columns={"DecLat": "lat_o", "DecLon": "lon_o"}).drop(columns=["Airport"])

    df_map = df_map.merge(coords, left_on="Destination IATA", right_on="Airport", how="left")
    df_map = df_map.rename(columns={"DecLat": "lat_d", "DecLon": "lon_d"}).drop(columns=["Airport"])

    # Remove rows with missing coords
    df_map = df_map.dropna(subset=["lat_o", "lon_o", "lat_d", "lon_d"])

    kepler_data = df_map[[
        "Origin Airport", "Destination Airport",
        "lat_o", "lon_o", "lat_d", "lon_d",
        "Passenger Δ (%)"
    ]]

    geojson_config = {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [{
                    "type": "arc",
                    "config": {
                        "dataId": "traffic",
                        "label": "Traffic Δ",
                        "color": [255, 0, 0],
                        "columns": {
                            "lat0": "lat_o", "lng0": "lon_o",
                            "lat1": "lat_d", "lng1": "lon_d"
                        },
                        "isVisible": True,
                        "sizeField": "Passenger Δ (%)",
                        "sizeScale": 1
                    }
                }]
            }
        }
    }

    map_ = KeplerGl(height=500)
    map_.add_data(data=kepler_data, name="traffic")
    map_.config = geojson_config

    st.subheader("🌍 Traffic Flow Map (Kepler)")
    components.html(map_._repr_html_(), height=600)
else:
    st.warning("📍 Upload a valid Excel with IATA_Code, DecLat, and DecLon to view the map.")

# ----------------------
# Summary Metrics
# ----------------------
col1, col2 = st.columns(2)
with col1:
    base = df["Passengers"].sum()
    new = df["Passengers after policy"].sum()
    st.metric("Total Passengers (M)", f"{new / 1e6:.2f}", f"{(new / base - 1)*100:+.1f}%")

with col2:
    st.metric("Avg. Carbon Cost (€)", f"{df['Carbon cost per pax'].mean():.2f}")
