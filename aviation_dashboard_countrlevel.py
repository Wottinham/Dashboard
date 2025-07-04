import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl

# Defaults
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND = 1.4

# Load dummy data
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

# Streamlit UI
st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")

st.sidebar.header("📈 Policy & Economic Inputs")
uploaded_file = st.sidebar.file_uploader("Upload airport-pair passenger CSV", type=["csv"])
coord_file = st.sidebar.file_uploader("Upload airport coordinates Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded.")
else:
    df = load_dummy_data()
    st.info("🛈 No file uploaded – using dummy data.")

expected_columns = {"Origin Country Name", "Destination Country Name", "Origin Airport", "Destination Airport", "Distance (km)", "Passengers", "Avg. Total Fare(USD)"}
if not expected_columns.issubset(df.columns):
    st.error("CSV missing required columns.")
    st.stop()
df = df.dropna()

origin_all = sorted(df["Origin Country Name"].unique())
dest_all = sorted(df["Destination Country Name"].unique())

# Carbon price input
st.sidebar.markdown("### Carbon Pricing Policy")
carbon_policy = st.sidebar.selectbox("Enable carbon pricing?", ["Disable", "Enable"])
if carbon_policy == "Enable":
    ets_price = st.sidebar.slider("Carbon price (EUR / tCO₂)", 0, 400, 100, 5)
    carbon_origin_countries = st.sidebar.multiselect("Origin countries taxed:", origin_all, default=origin_all)
    carbon_dest_countries = st.sidebar.multiselect("Destination countries taxed:", dest_all, default=dest_all)
else:
    ets_price = 0.0
    carbon_origin_countries = []
    carbon_dest_countries = []

# Air Passenger Tax
st.sidebar.markdown("### Passenger Tax")
tax_policy = st.sidebar.selectbox("Enable air passenger tax?", ["Disable", "Enable"])
if tax_policy == "Enable":
    air_passenger_tax = st.sidebar.slider("Air Passenger Tax (USD)", 0, 100, 10, 1)
    tax_origin_countries = st.sidebar.multiselect("Origin countries taxed:", origin_all, default=origin_all)
    tax_dest_countries = st.sidebar.multiselect("Destination countries taxed:", dest_all, default=dest_all)
else:
    air_passenger_tax = 0.0
    tax_origin_countries = []
    tax_dest_countries = []

# Parameters
st.sidebar.markdown("### Parameters")
pass_through = st.sidebar.slider("Cost pass-through to fares (%)", 0, 100, 80, 5) / 100
emission_factor = st.sidebar.slider("Emission factor (kg CO₂ per pax-km)", 0.0, 1.0, 0.115, 0.001)

# GDP & elasticity
global_gdp_growth = st.sidebar.slider("Global Real GDP growth (%)", -5.0, 8.0, 2.5, 0.1)
user_price_elasticity = st.sidebar.slider("Price-elasticity (negative)", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1)
user_gdp_elasticity = st.sidebar.slider("GDP-elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1)

# Optional per-country GDP
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth by Country"):
    for country in origin_all:
        gdp_growth_by_country[country] = st.slider(
            f"{country} GDP growth (%)", -5.0, 8.0, global_gdp_growth, 0.1, key=f"gdp_{country}"
        )

# Carbon cost and tax
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor
df["Carbon cost per pax"] = 0.0
if carbon_policy == "Enable":
    mask_c = df["Origin Country Name"].isin(carbon_origin_countries) & df["Destination Country Name"].isin(carbon_dest_countries)
    df.loc[mask_c, "Carbon cost per pax"] = (df.loc[mask_c, "CO2 per pax (kg)"] / 1000) * ets_price * pass_through

df["Air passenger tax per pax"] = 0.0
if tax_policy == "Enable":
    mask_t = df["Origin Country Name"].isin(tax_origin_countries) & df["Destination Country Name"].isin(tax_dest_countries)
    df.loc[mask_t, "Air passenger tax per pax"] = air_passenger_tax * pass_through

# Fare & Demand
df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"] + df["Air passenger tax per pax"]
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100
fare_factor = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"]).replace([np.inf, -np.inf], np.nan) ** user_price_elasticity
df["GDP Growth (%)"] = df["Origin Country Name"].map(gdp_growth_by_country).fillna(global_gdp_growth)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"] / 100) ** user_gdp_elasticity
df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# Optional: map coordinates
airport_coords = None
if coord_file:
    try:
        coords_df = pd.read_excel(coord_file, engine="openpyxl")
        if {"IATA_Code", "DecLat", "DecLon"}.issubset(coords_df.columns):
            airport_coords = coords_df.set_index("IATA_Code")[["DecLat", "DecLon"]]
            df["Origin Lat"] = df["Origin Airport"].str[:3].map(airport_coords["DecLat"])
            df["Origin Lon"] = df["Origin Airport"].str[:3].map(airport_coords["DecLon"])
            df["Dest Lat"] = df["Destination Airport"].str[:3].map(airport_coords["DecLat"])
            df["Dest Lon"] = df["Destination Airport"].str[:3].map(airport_coords["DecLon"])
    except Exception as e:
        st.warning(f"❌ Failed to process coordinate file: {e}")

# Output – Results Table
st.subheader("📊 Passenger Simulation Results (Airport-Pair Level)")
st.dataframe(df[[
    "Origin Airport", "Destination Airport",
    "Passengers", "Avg. Total Fare(USD)",
    "Carbon cost per pax", "Air passenger tax per pax",
    "New Avg Fare", "Passenger Δ (%)"
]], use_container_width=True)

# Output – Bar Chart
origin_summary = df.groupby("Origin Country Name", as_index=False).agg({
    "Passengers": "sum", "Passengers after policy": "sum"
})
origin_summary["Relative Change (%)"] = (
    origin_summary["Passengers after policy"] / origin_summary["Passengers"] - 1
) * 100

fig = px.bar(origin_summary, x="Origin Country Name", y="Relative Change (%)",
             title="📉 Relative Change in Passenger Volume by Origin Country",
             labels={"Relative Change (%)": "Change in Passengers (%)"}, text="Relative Change (%)")
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

# Output – Metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Passengers (m)", f"{df['Passengers after policy'].sum()/1e6:,.2f} M",
              delta=f"{(df['Passengers after policy'].sum()/df['Passengers'].sum()-1)*100:+.1f}%")
with col2:
    st.metric("Avg. Carbon Cost (€)", f"{df['Carbon cost per pax'].mean():.2f}")

# Output – Kepler Map
if airport_coords is not None:
    kepler_df = df.dropna(subset=["Origin Lat", "Origin Lon", "Dest Lat", "Dest Lon"]).copy()
    kepler_df["origin_lat"] = kepler_df["Origin Lat"]
    kepler_df["origin_lng"] = kepler_df["Origin Lon"]
    kepler_df["dest_lat"] = kepler_df["Dest Lat"]
    kepler_df["dest_lng"] = kepler_df["Dest Lon"]
    kepler_df["traffic_change"] = kepler_df["Passenger Δ (%)"]

    map_ = KeplerGl(height=600)
    map_.add_data(data=kepler_df, name="Air Traffic Change")
    st.subheader("🌍 Air Traffic Change Map")
    keplergl_static(map_)
