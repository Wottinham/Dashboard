import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static

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
                "Passengers": int(rng.integers(50000, 1000000)),
                "Avg. Total Fare(USD)": round(rng.uniform(150, 700), 2),
                "Origin Lat": rng.uniform(-50, 60),
                "Origin Lon": rng.uniform(-130, 130),
                "Destination Lat": rng.uniform(-50, 60),
                "Destination Lon": rng.uniform(-130, 130),
            })
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between airports and policy impacts.")

st.sidebar.header("📈 Policy & Economic Inputs")
uploaded_file = st.sidebar.file_uploader("Upload airport-pair passenger CSV", type=["csv"])

# ----------------------
# Load data – CSV if provided, otherwise dummy
# ----------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded.")
else:
    df = load_dummy_data()
    st.info("🛈 No file uploaded – using **dummy data** to explore the simulator.")

# Validate columns
expected_columns = {
    "Origin Country Name", "Destination Country Name",
    "Origin Airport", "Destination Airport",
    "Distance (km)", "Passengers", "Avg. Total Fare(USD)"
}
if not expected_columns.issubset(df.columns):
    st.error(
        "CSV must contain columns: Origin Country Name, Destination Country Name, "
        "Origin Airport, Destination Airport, Distance (km), Passengers, Avg. Total Fare(USD)"
    )
    st.stop()
df = df.dropna(subset=expected_columns)
df["Avg. Total Fare(USD)"] = df["Avg. Total Fare(USD)"].fillna(0.0)

origin_all = sorted(df["Origin Country Name"].unique())
dest_all = sorted(df["Destination Country Name"].unique())

# ----------------------
# Carbon Pricing Policy
# ----------------------
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

# ----------------------
# Tax
# ----------------------
st.sidebar.markdown("### Passenger Tax")
tax_policy = st.sidebar.selectbox("Enable air passenger tax?", ["Disable", "Enable"])
if tax_policy == "Enable":
    air_passenger_tax = st.sidebar.slider("Air Passenger Tax (USD)", 0, 100, 0, 1)
    tax_origin_countries = st.sidebar.multiselect("Origin countries taxed:", origin_all, default=origin_all)
    tax_dest_countries = st.sidebar.multiselect("Destination countries taxed:", dest_all, default=dest_all)
else:
    air_passenger_tax = 0.0
    tax_origin_countries = []
    tax_dest_countries = []

# ----------------------
# Parameters
# ----------------------
st.sidebar.markdown("### Parameters")

pass_through = st.sidebar.slider(
    "Cost pass-through to fares (%)", 0, 100, 80, 5
) / 100

emission_factor = st.sidebar.slider(
    "Emission factor (kg CO₂ per pax-km)", 0.0, 1.0, 0.115, 0.001
)

# ----------------------
# Other Economic Inputs
# ----------------------
global_gdp_growth = st.sidebar.slider("Global Real GDP growth (%)", -5.0, 8.0, 2.5, 0.1)
user_price_elasticity = st.sidebar.slider("Demand price-elasticity", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1)
user_gdp_elasticity = st.sidebar.slider("Demand GDP-elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1)

st.sidebar.markdown("### Optional: Adjust GDP Growth by Origin Country")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth for Specific Countries"):
    for country in origin_all:
        gdp_growth_by_country[country] = st.slider(
            f"{country} GDP growth (%)", -5.0, 8.0, global_gdp_growth, 0.1, key=f"gdp_{country}"
        )

# ----------------------
# Policy calculations
# ----------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor
df["Carbon cost per pax"] = 0.0

if carbon_policy == "Enable":
    mask_c = (
        df["Origin Country Name"].isin(carbon_origin_countries) &
        df["Destination Country Name"].isin(carbon_dest_countries)
    )
    df.loc[mask_c, "Carbon cost per pax"] = (
        df.loc[mask_c, "CO2 per pax (kg)"] / 1000 * ets_price * pass_through
    )

df["Air passenger tax per pax"] = 0.0
if tax_policy == "Enable":
    mask_t = (
        df["Origin Country Name"].isin(tax_origin_countries) &
        df["Destination Country Name"].isin(tax_dest_countries)
    )
    df.loc[mask_t, "Air passenger tax per pax"] = air_passenger_tax * pass_through

df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"] + df["Air passenger tax per pax"]
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

fare_factor = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"]).replace([np.inf, -np.inf], np.nan) ** user_price_elasticity
df["GDP Growth (%)"] = df["Origin Country Name"].map(gdp_growth_by_country).fillna(global_gdp_growth)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"] / 100) ** user_gdp_elasticity

df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# ----------------------
# Output Table
# ----------------------
st.subheader("📊 Passenger Simulation Results (Airport-Pair Level)")
st.dataframe(df[[
    "Origin Airport", "Destination Airport",
    "Origin Country Name", "Destination Country Name",
    "Passengers", "Distance (km)", "CO2 per pax (kg)",
    "Avg. Total Fare(USD)", "Carbon cost per pax", "Air passenger tax per pax",
    "New Avg Fare", "Passengers after policy", "Passenger Δ (%)"
]], use_container_width=True)

# ----------------------
# Bar Chart
# ----------------------
origin_summary = df.groupby("Origin Country Name", as_index=False).agg({
    "Passengers": "sum",
    "Passengers after policy": "sum",
    "CO2 per pax (kg)": "mean",
})
origin_summary["Relative Change (%)"] = (
    origin_summary["Passengers after policy"] / origin_summary["Passengers"] - 1
) * 100

fig = px.bar(
    origin_summary,
    x="Origin Country Name", y="Relative Change (%)",
    title="Relative Change in Passenger Volume by Origin Country (%)"
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Metrics
# ----------------------
col1, col2 = st.columns(2)
with col1:
    base_total = df["Passengers"].sum()
    new_total = df["Passengers after policy"].sum()
    delta_pct = (new_total / base_total - 1) * 100
    st.metric("Total Passengers (m)", f"{new_total/1e6:,.2f} M", delta=f"{delta_pct:+.1f}% vs base")
with col2:
    avg_cc = df["Carbon cost per pax"].mean()
    st.metric("Avg Carbon Cost per Ticket", f"€{avg_cc:.2f}")


# ----------------------
# Optional: Upload coordinates for airports
# ----------------------
st.sidebar.markdown("### Optional: Upload Airport Coordinates")
coord_file = st.sidebar.file_uploader("Upload airport coordinates (xlsx)", type=["xlsx"])

iata_coords = None
if coord_file is not None:
    try:
        coords_df = pd.read_excel(coord_file)
        if {'IATA_Code', 'DecLat', 'DecLon'}.issubset(coords_df.columns):
            iata_coords = coords_df.set_index('IATA_Code')[['DecLat', 'DecLon']]
            st.success("✅ Airport coordinates loaded.")

            # Match for Origin
            df = df.merge(iata_coords, how='left', left_on='Origin Airport', right_index=True)
            df.rename(columns={'DecLat': 'Origin Lat', 'DecLon': 'Origin Lon'}, inplace=True)

            # Match for Destination
            df = df.merge(iata_coords, how='left', left_on='Destination Airport', right_index=True)
            df.rename(columns={'DecLat': 'Destination Lat', 'DecLon': 'Destination Lon'}, inplace=True)

            # Drop rows with missing coordinates
            df.dropna(subset=['Origin Lat', 'Origin Lon', 'Destination Lat', 'Destination Lon'], inplace=True)
        else:
            st.error("❌ The coordinate file must contain: IATA_Code, DecLat, DecLon.")
    except Exception as e:
        st.error(f"❌ Failed to process coordinate file: {e}")
        
# ----------------------
# Kepler Map
# ----------------------
st.subheader("🗺️ Air Traffic Flow Changes (Kepler Map)")
kepler_data = df[[
    "Origin Lon", "Origin Lat", "Destination Lon", "Destination Lat", "Passenger Δ (%)"
]].rename(columns={
    "Origin Lon": "longitude", "Origin Lat": "latitude",
    "Destination Lon": "to_longitude", "Destination Lat": "to_latitude"
})
kepler_data["Passenger Δ (%)"] = df["Passenger Δ (%)"]

kepler_map = KeplerGl(height=600)
kepler_map.add_data(data=kepler_data, name="Air Traffic")
keplergl_static(kepler_map)

# ----------------------
# Footer
# ----------------------
st.info("💡 Each country inherits the global GDP growth unless adjusted manually.")
st.caption("Data: Sabre MI (or dummy) · Visualization powered by Streamlit, Plotly, and Kepler.gl")
