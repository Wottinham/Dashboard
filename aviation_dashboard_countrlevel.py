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
GDP_ELASTICITY_DEMAND   = 1.4

# ----------------------
# Helper – generate dummy data when no CSV is provided
# ----------------------
def load_dummy_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    origins = ["Germany", "France", "United States", "Japan"]
    dests   = ["Spain", "Italy", "United Kingdom", "Canada"]
    rows = []
    for o in origins:
        for d in dests:
            if o == d:
                continue
            rows.append({
                "Origin Country Name":      o,
                "Destination Country Name": d,
                "Origin Airport":           f"{o[:3].upper()}-INTL",
                "Destination Airport":      f"{d[:3].upper()}-INTL",
                "Distance (km)":            int(rng.integers(500, 9000)),
                "Passengers":               int(rng.integers(50_000, 1_000_000)),
                "Avg. Total Fare(USD)":     round(rng.uniform(150, 700), 2),
            })
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI setup
# ----------------------
st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between airports and policy impacts.")

# Sidebar – uploads
st.sidebar.header("📈 Policy & Data Inputs")
uploaded_file = st.sidebar.file_uploader(
    "Upload airport-pair passenger CSV", type=["csv"], key="upload_csv"
)
coord_file = st.sidebar.file_uploader(
    "Upload airport coordinates (.xlsx)", type=["xlsx"], key="upload_coords"
)

# ----------------------
# Load passenger data
# ----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Passenger CSV loaded.")
else:
    df = load_dummy_data()
    st.info("🛈 No passenger CSV – using **dummy data**.")

# Validate passenger columns
required_cols = {
    "Origin Country Name", "Destination Country Name",
    "Origin Airport",        "Destination Airport",
    "Distance (km)",         "Passengers",
    "Avg. Total Fare(USD)"
}
if not required_cols.issubset(df.columns):
    st.error("Passenger CSV missing required columns.")
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
        "Origin countries taxed (carbon):",
        origin_all, default=origin_all, key="mslt_carbon_orig"
    )
    carbon_dest_countries = st.sidebar.multiselect(
        "Destination countries taxed (carbon):",
        dest_all, default=dest_all, key="mslt_carbon_dest"
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
        "Origin countries taxed (tax):",
        origin_all, default=origin_all, key="mslt_tax_orig"
    )
    tax_dest_countries = st.sidebar.multiselect(
        "Destination countries taxed (tax):",
        dest_all, default=dest_all, key="mslt_tax_dest"
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
    "Cost pass-through to fares (%)", 0, 100, 80, 5,
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
    "Global real GDP growth (%)", -5.0, 8.0, 2.5, 0.1,
    key="slider_gdp_global"
)
user_price_elasticity = st.sidebar.slider(
    "Demand price elasticity (negative)",
    -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1,
    key="slider_price_elast"
)
user_gdp_elasticity = st.sidebar.slider(
    "Demand GDP elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1,
    key="slider_gdp_elast"
)

st.sidebar.markdown("### Optional: Adjust GDP Growth by Country")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth for Specific Origins"):
    for country in origin_all:
        gdp_growth_by_country[country] = st.slider(
            f"{country} GDP growth (%)", -5.0, 8.0, global_gdp_growth, 0.1,
            key=f"gdp_{country}"
        )

# ----------------------
# Policy calculations
# ----------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor

# Carbon cost
df["Carbon cost per pax"] = 0.0
if enable_carbon:
    mask_c = (
        df["Origin Country Name"].isin(carbon_origin_countries) &
        df["Destination Country Name"].isin(carbon_dest_countries)
    )
    df.loc[mask_c, "Carbon cost per pax"] = (
        df.loc[mask_c, "CO2 per pax (kg)"] / 1_000
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

# New fare
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
# Initialize coords so dropna never KeyErrors
# ----------------------
df["Origin Lat"] = np.nan
df["Origin Lon"] = np.nan
df["Dest Lat"]   = np.nan
df["Dest Lon"]   = np.nan

# ----------------------
# Load & merge coordinates
# ----------------------
if coord_file:
    try:
        coords_df = pd.read_excel(coord_file, engine="openpyxl")
        coords_df = coords_df.drop_duplicates(subset=["IATA_Code"])
        if {"IATA_Code", "DecLat", "DecLon"}.issubset(coords_df.columns):
            coords_map = coords_df.set_index("IATA_Code")[["DecLat", "DecLon"]]
            # use .partition to grab code before the dash
            df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
            df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
            df["Origin Lat"]  = df["Origin Code"].map(coords_map["DecLat"])
            df["Origin Lon"]  = df["Origin Code"].map(coords_map["DecLon"])
            df["Dest Lat"]    = df["Dest Code"].map(coords_map["DecLat"])
            df["Dest Lon"]    = df["Dest Code"].map(coords_map["DecLon"])
        else:
            st.warning("❌ Coordinate file missing IATA_Code / DecLat / DecLon.")
    except ImportError:
        st.warning("❌ Install 'openpyxl' to read .xlsx files: `pip install openpyxl`.")
    except Exception as e:
        st.warning(f"❌ Failed to process coordinate file: {e}")

# ----------------------
# Table & bar chart outputs
# ----------------------
st.subheader("📊 Airport-Pair Passenger Results")
st.dataframe(
    df[[
        "Origin Airport", "Destination Airport", "Passengers",
        "Distance (km)", "CO2 per pax (kg)",
        "Avg. Total Fare(USD)", "Carbon cost per pax",
        "Air passenger tax per pax", "New Avg Fare",
        "Passenger Δ (%)"
    ]],
    use_container_width=True
)

# barplot by origin country
origin_summary = df.groupby("Origin Country Name", as_index=False).agg({
    "Passengers":               "sum",
    "Passengers after policy":  "sum"
})
origin_summary["Relative Change (%)"] = (
    origin_summary["Passengers after policy"] /
    origin_summary["Passengers"] - 1
) * 100

fig = px.bar(
    origin_summary,
    x="Origin Country Name",
    y="Relative Change (%)",
    title="📉 Relative Change in Passenger Volume by Origin Country",
    text="Relative Change (%)",
    labels={"Relative Change (%)": "Δ Passengers (%)"}
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

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



# ----------------------
# Kepler map: country-level arcs with color ramp
# ----------------------
# compute centroids as before…
origin_centroids = (
    df.groupby("Origin Country Name")[["Origin Lat", "Origin Lon"]]
      .mean()
      .reset_index()
)
dest_centroids = (
    df.groupby("Destination Country Name")[["Dest Lat", "Dest Lon"]]
      .mean()
      .reset_index()
)

country_agg = (
    df.groupby(["Origin Country Name", "Destination Country Name"], as_index=False)
      .agg({"Passengers": "sum", "Passengers after policy": "sum"})
)
country_agg["traffic_change"] = (
    country_agg["Passengers after policy"] / country_agg["Passengers"] - 1
) * 100

country_agg = (
    country_agg
    .merge(origin_centroids, on="Origin Country Name", how="left")
    .merge(dest_centroids,   on="Destination Country Name", how="left")
)

kepler_df = country_agg.dropna(subset=["Origin Lat", "Origin Lon", "Dest Lat", "Dest Lon"])
if not kepler_df.empty:
    st.subheader("🌍 Air Traffic Change Map (Country-Level)")

    kepler_df = kepler_df.rename(columns={
        "Origin Lat": "origin_lat",
        "Origin Lon": "origin_lng",
        "Dest Lat":   "dest_lat",
        "Dest Lon":   "dest_lng",
        "Origin Country Name":      "origin_country",
        "Destination Country Name": "dest_country",
    })

    kepler_config = {
      "version": "v1",
      "config": {
        "visState": {
          "layers": [{
            "id": "country-traffic-arcs",
            "type": "arc",
            "config": {
              "dataId": "Country Traffic Change",
              "label": "Traffic Δ Arcs",
              "columns": {
                "lat0": "origin_lat",
                "lng0": "origin_lng",
                "lat1": "dest_lat",
                "lng1": "dest_lng"
              },
              "isVisible": True,
              "visConfig": {
                "opacity": 0.9,
                "thickness": 4,
                "colorField": {"name": "traffic_change", "type": "real"},
                "colorScale": "quantile",
                "colorRange": {
                  "name": "Traffic Change",
                  "type": "sequential",
                  "category": "Uber",
                  "colors": ["#2ca25f", "#fee08b", "#f03b20"]
                }
              }
            }
          }],
          "interactionConfig": {
            "tooltip": {
              "fieldsToShow": {
                "Country Traffic Change": [
                  {"name": "origin_country"},
                  {"name": "dest_country"},
                  {"name": "traffic_change"}
                ]
              }
            }
          },
          "layerBlending": "normal"
        }
      }
    }

    m = KeplerGl(config=kepler_config, height=800)  # ↑ bigger map height
    m.add_data(data=kepler_df, name="Country Traffic Change")
    keplergl_static(m)
