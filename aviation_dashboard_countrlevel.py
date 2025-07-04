import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl

# ----------------------
# Model configuration (defaults)
# ----------------------
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
            rows.append({
                "Origin Country Name": o,
                "Destination Country Name": d,
                "Origin Airport": f"{o[:3].upper()}-INTL",
                "Destination Airport": f"{d[:3].upper()}-INTL",
                "Distance (km)": int(rng.integers(500, 9000)),
                "Passengers": int(rng.integers(50_000, 1_000_000)),
                "Avg. Total Fare(USD)": round(rng.uniform(150, 700), 2),
            })
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI setup
# ----------------------
st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between airports and policy impacts.")

# Sidebar uploads
st.sidebar.header("📈 Policy & Data Inputs")
uploaded_file = st.sidebar.file_uploader("Upload airport-pair passenger CSV", type=["csv"], key="upload_csv")
coord_file   = st.sidebar.file_uploader("Upload airport coordinates (.xlsx)", type=["xlsx"], key="upload_coords")

# ----------------------
# Load passenger data
# ----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded.")
else:
    df = load_dummy_data()
    st.info("🛈 No file uploaded – using **dummy data**.")

# Validate
required_cols = {
    "Origin Country Name", "Destination Country Name",
    "Origin Airport", "Destination Airport",
    "Distance (km)", "Passengers", "Avg. Total Fare(USD)"
}
if not required_cols.issubset(df.columns):
    st.error("CSV missing required columns.")
    st.stop()
df = df.dropna(subset=required_cols).reset_index(drop=True)

origin_all = sorted(df["Origin Country Name"].unique())
dest_all   = sorted(df["Destination Country Name"].unique())

# ----------------------
# Carbon pricing policy
# ----------------------
enable_carbon = st.sidebar.checkbox("Enable carbon pricing?", key="chk_carbon")
if enable_carbon:
    ets_price = st.sidebar.slider(
        "Carbon price (EUR / tCO₂)", 0, 400, 100, 5, key="slider_ets"
    )
    carbon_origin_countries = st.sidebar.multiselect(
        "Origin countries taxed (carbon):", origin_all,
        default=origin_all, key="mslt_carbon_orig"
    )
    carbon_dest_countries = st.sidebar.multiselect(
        "Destination countries taxed (carbon):", dest_all,
        default=dest_all, key="mslt_carbon_dest"
    )
else:
    ets_price = 0.0
    carbon_origin_countries = []
    carbon_dest_countries = []

# ----------------------
# Air passenger tax policy
# ----------------------
enable_tax = st.sidebar.checkbox("Enable air passenger tax?", key="chk_tax")
if enable_tax:
    air_passenger_tax = st.sidebar.slider(
        "Air Passenger Tax (USD)", 0, 100, 10, 1, key="slider_tax"
    )
    tax_origin_countries = st.sidebar.multiselect(
        "Origin countries taxed (tax):", origin_all,
        default=origin_all, key="mslt_tax_orig"
    )
    tax_dest_countries = st.sidebar.multiselect(
        "Destination countries taxed (tax):", dest_all,
        default=dest_all, key="mslt_tax_dest"
    )
else:
    air_passenger_tax = 0.0
    tax_origin_countries = []
    tax_dest_countries = []

# ----------------------
# Parameters
# ----------------------
st.sidebar.markdown("### Parameters")
pass_through = st.sidebar.slider(
    "Cost pass-through to fares (%)",
    0, 100, 80, 5,
    help="Share of carbon cost and ticket tax airlines embed in ticket prices.",
    key="slider_pass_through"
) / 100

emission_factor = st.sidebar.slider(
    "Emission factor (kg CO₂ per pax-km)",
    0.0, 1.0, 0.115, 0.001,
    help="kg of CO₂ emitted per passenger-km flown.",
    key="slider_emission_factor"
)

# ----------------------
# Economic inputs
# ----------------------
global_gdp_growth = st.sidebar.slider(
    "Global real GDP growth (%)",
    -5.0, 8.0, 2.5, 0.1,
    key="slider_gdp_global"
)
user_price_elasticity = st.sidebar.slider(
    "Demand price elasticity (negative)",
    -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1,
    key="slider_price_elast"
)
user_gdp_elasticity = st.sidebar.slider(
    "Demand GDP elasticity",
    0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1,
    key="slider_gdp_elast"
)

st.sidebar.markdown("### Optional: Adjust GDP Growth by Country")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth for Specific Origins"):
    for country in origin_all:
        gdp_growth_by_country[country] = st.slider(
            f"{country} GDP growth (%)",
            -5.0, 8.0, global_gdp_growth, 0.1,
            key=f"gdp_{country}"
        )

# ----------------------
# Policy calculations
# ----------------------
# Emissions
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor

# Carbon cost
df["Carbon cost per pax"] = 0.0
if enable_carbon:
    mask_c = (
        df["Origin Country Name"].isin(carbon_origin_countries) &
        df["Destination Country Name"].isin(carbon_dest_countries)
    )
    df.loc[mask_c, "Carbon cost per pax"] = (
        df.loc[mask_c, "CO2 per pax (kg)"] / 1000
        * ets_price * pass_through
    )

# Passenger tax
df["Air passenger tax per pax"] = 0.0
if enable_tax:
    mask_t = (
        df["Origin Country Name"].isin(tax_origin_countries) &
        df["Destination Country Name"].isin(tax_dest_countries)
    )
    df.loc[mask_t, "Air passenger tax per pax"] = air_passenger_tax * pass_through

# New fare & fare change
df["New Avg Fare"] = (
    df["Avg. Total Fare(USD)"]
    + df["Carbon cost per pax"]
    + df["Air passenger tax per pax"]
)
df["Fare Δ (%)"] = (
    df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1
) * 100

# Elasticity & GDP
fare_factor = (
    (df["New Avg Fare"] / df["Avg. Total Fare(USD)"])
    .replace([np.inf, -np.inf], np.nan) ** user_price_elasticity
)
df["GDP Growth (%)"] = (
    df["Origin Country Name"]
    .map(gdp_growth_by_country)
    .fillna(global_gdp_growth)
)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"] / 100) ** user_gdp_elasticity

df["Passengers after policy"] = (
    df["Passengers"] * fare_factor * df["GDP Growth Factor"]
)
df["Passenger Δ (%)"] = (
    df["Passengers after policy"] / df["Passengers"] - 1
) * 100

# ----------------------
# Optional: load & join coords
# ----------------------
if coord_file:
    try:
        coords_df = pd.read_excel(coord_file, engine="openpyxl")
        if {"IATA_Code", "DecLat", "DecLon"}.issubset(coords_df.columns):
            coords_map = coords_df.set_index("IATA_Code")[["DecLat", "DecLon"]]
            # extract code from "XXX-INTL"
            df["Origin_Code"] = df["Origin Airport"].str.split("-", 1).str[0]
            df["Dest_Code"]   = df["Destination Airport"].str.split("-", 1).str[0]
            df["Origin Lat"] = df["Origin_Code"].map(coords_map["DecLat"])
            df["Origin Lon"] = df["Origin_Code"].map(coords_map["DecLon"])
            df["Dest Lat"]   = df["Dest_Code"].map(coords_map["DecLat"])
            df["Dest Lon"]   = df["Dest_Code"].map(coords_map["DecLon"])
        else:
            st.warning("❌ Coordinate file missing IATA_Code/DecLat/DecLon columns.")
    except Exception as e:
        st.warning(f"❌ Failed to process coordinate file: {e}")

# ----------------------
# OUTPUTS
# ----------------------
st.subheader("📊 Airport-Pair Passenger Results")
st.dataframe(
    df[[
        "Origin Airport", "Destination Airport",
        "Passengers", "Distance (km)", "CO2 per pax (kg)",
        "Avg. Total Fare(USD)", "Carbon cost per pax",
        "Air passenger tax per pax", "New Avg Fare",
        "Passenger Δ (%)"
    ]],
    use_container_width=True
)

# Barplot by origin country
origin_summary = df.groupby("Origin Country Name", as_index=False).agg({
    "Passengers": "sum",
    "Passengers after policy": "sum",
})
origin_summary["Relative Change (%)"] = (
    origin_summary["Passengers after policy"]
    / origin_summary["Passengers"] - 1
) * 100

fig = px.bar(
    origin_summary,
    x="Origin Country Name",
    y="Relative Change (%)",
    title="📉 Relative Change in Passenger Volume by Origin Country",
    labels={"Relative Change (%)": "Δ Passengers (%)"},
    text="Relative Change (%)"
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# Metrics
col1, col2 = st.columns(2)
with col1:
    total_base = df["Passengers"].sum()
    total_new  = df["Passengers after policy"].sum()
    st.metric(
        "Total Passengers (M)",
        f"{total_new/1e6:,.2f}",
        delta=f"{(total_new/total_base-1)*100:+.1f}%"
    )
with col2:
    st.metric(
        "Avg. Carbon Cost (€)",
        f"{df['Carbon cost per pax'].mean():.2f}"
    )

st.info("💡 Each country inherits the global GDP growth unless adjusted manually.")
st.caption("Data: Sabre MI (or dummy) · Visualization by Streamlit & Plotly")

# Kepler map (only if all four coord columns exist)
coords_required = {"Origin Lat", "Origin Lon", "Dest Lat", "Dest Lon"}
if coords_required.issubset(df.columns):
    st.subheader("🌍 Air Traffic Change Map")
    kepler_df = df.dropna(subset=list(coords_required)).copy()
    kepler_df["origin_lat"]   = kepler_df["Origin Lat"]
    kepler_df["origin_lng"]   = kepler_df["Origin Lon"]
    kepler_df["dest_lat"]     = kepler_df["Dest Lat"]
    kepler_df["dest_lng"]     = kepler_df["Dest Lon"]
    kepler_df["traffic_change"] = kepler_df["Passenger Δ (%)"]

    m = KeplerGl(height=600)
    m.add_data(data=kepler_df, name="Air Traffic Change")
    keplergl_static(m)
