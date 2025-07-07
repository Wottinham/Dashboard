import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from keplergl import KeplerGl
import streamlit.components.v1 as components

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
uploaded_file = st.sidebar.file_uploader("Upload passenger CSV", type=["csv"])
coord_file    = st.sidebar.file_uploader("Upload coords (.xlsx)", type=["xlsx"])

# ----------------------
# Load passenger data
# ----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Passenger CSV loaded.")
else:
    df = load_dummy_data()
    st.info("🛈 No passenger CSV – using **dummy data**.")

# Validate required columns
required_cols = {
    "Origin Country Name", "Destination Country Name",
    "Origin Airport", "Destination Airport",
    "Distance (km)", "Passengers", "Avg. Total Fare(USD)"
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
enable_carbon = st.sidebar.checkbox("Enable carbon pricing?")
if enable_carbon:
    ets_price = st.sidebar.slider("Carbon price (EUR/tCO₂)", 0,400,100,5)
    carbon_origin_countries = st.sidebar.multiselect(
        "Carbon: Origin countries", origin_all, default=origin_all)
    carbon_dest_countries   = st.sidebar.multiselect(
        "Carbon: Dest countries", dest_all, default=dest_all)
else:
    ets_price = 0.0
    carbon_origin_countries = []
    carbon_dest_countries   = []

# ----------------------
# Air passenger tax policy
# ----------------------
enable_tax = st.sidebar.checkbox("Enable air passenger tax?")
if enable_tax:
    air_passenger_tax = st.sidebar.slider("Air Passenger Tax (USD)", 0,100,10,1)
    tax_origin_countries = st.sidebar.multiselect(
        "Tax: Origin countries", origin_all, default=origin_all)
    tax_dest_countries   = st.sidebar.multiselect(
        "Tax: Dest countries", dest_all, default=dest_all)
else:
    air_passenger_tax = 0.0
    tax_origin_countries = []
    tax_dest_countries   = []

# ----------------------
# Parameters
# ----------------------
st.sidebar.markdown("### Parameters")
pass_through = st.sidebar.slider(
    "Cost pass-through (%)", 0,100,80,5) / 100
emission_factor = st.sidebar.slider(
    "Emission factor (kg CO₂/pax-km)", 0.0,1.0,0.115,0.001)

# ----------------------
# Economic inputs
# ----------------------
global_gdp_growth = st.sidebar.slider("Global GDP growth (%)", -5.0,8.0,2.5,0.1)
user_price_elast  = st.sidebar.slider(
    "Price elasticity", -2.0,-0.1,PRICE_ELASTICITY_DEMAND,0.1)
user_gdp_elast    = st.sidebar.slider(
    "GDP elasticity", 0.5,2.0,GDP_ELASTICITY_DEMAND,0.1)

st.sidebar.markdown("### Optional GDP by Origin")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth"):
    for c in origin_all:
        gdp_growth_by_country[c] = st.sidebar.slider(
            f"{c} GDP (%)", -5.0,8.0,global_gdp_growth,0.1,
            key=f"gdp_{c}"
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

# New fare & changes
df["New Avg Fare"] = (
    df["Avg. Total Fare(USD)"] +
    df["Carbon cost per pax"] +
    df["Air passenger tax per pax"]
)
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

# Elasticity & GDP
fare_factor = (
    (df["New Avg Fare"] / df["Avg. Total Fare(USD)"])
    .replace([np.inf,-np.inf],np.nan) ** user_price_elast
)
df["GDP Growth (%)"] = (
    df["Origin Country Name"].map(gdp_growth_by_country)
    .fillna(global_gdp_growth)
)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"]/100)**user_gdp_elast

df["Passengers after policy"] = (
    df["Passengers"] * fare_factor * df["GDP Growth Factor"]
)
df["Passenger Δ (%)"] = (
    df["Passengers after policy"] / df["Passengers"] - 1
) * 100

# Prepare coords columns
df["Origin Lat"] = np.nan; df["Origin Lon"] = np.nan
df["Dest Lat"]   = np.nan; df["Dest Lon"]   = np.nan

# ----------------------
# Load & merge coordinates
# ----------------------
if coord_file:
    try:
        coords_df = pd.read_excel(coord_file, engine="openpyxl")
        coords_df = coords_df.drop_duplicates("IATA_Code")
        if {"IATA_Code","DecLat","DecLon"}.issubset(coords_df.columns):
            cmap = coords_df.set_index("IATA_Code")[["DecLat","DecLon"]]
            df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
            df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
            df["Origin Lat"]  = df["Origin Code"].map(cmap["DecLat"])
            df["Origin Lon"]  = df["Origin Code"].map(cmap["DecLon"])
            df["Dest Lat"]    = df["Dest Code"].map(cmap["DecLat"])
            df["Dest Lon"]    = df["Dest Code"].map(cmap["DecLon"])
        else:
            st.warning("Coords missing IATA_Code/DecLat/DecLon.")
    except ImportError:
        st.warning("Install openpyxl: `pip install openpyxl`.")
    except Exception as e:
        st.warning(f"Failed to process coords: {e}")

# ----------------------
# Table & bar charts
# ----------------------
st.subheader("📊 Airport-Pair Passenger Results")
st.dataframe(df[[
    "Origin Airport","Destination Airport","Passengers",
    "Distance (km)","CO2 per pax (kg)","Avg. Total Fare(USD)",
    "Carbon cost per pax","Air passenger tax per pax",
    "New Avg Fare","Passenger Δ (%)"
]], use_container_width=True)

# 1) Passenger Δ bar
origin_sum = df.groupby("Origin Country Name", as_index=False).agg({
    "Passengers":"sum","Passengers after policy":"sum"
})
origin_sum["Δ Passengers (%)"] = (
    origin_sum["Passengers after policy"]/origin_sum["Passengers"] - 1
)*100

fig1 = px.bar(origin_sum,
              x="Origin Country Name", y="Δ Passengers (%)",
              title="📉 Δ Passenger Volume by Origin Country",
              text="Δ Passengers (%)")
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig1, use_container_width=True)

# 2) Fare Δ bar
fare_sum = df.groupby("Origin Country Name", as_index=False).agg({
    "Fare Δ (%)":"mean"
}).rename(columns={"Fare Δ (%)":"Avg Δ Fare (%)"})
fig2 = px.bar(fare_sum,
              x="Origin Country Name", y="Avg Δ Fare (%)",
              title="📈 Δ Average Fare by Origin Country",
              text="Avg Δ Fare (%)")
fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

# 3) Density curves of passenger distances
df_before = df[["Distance (km)","Passengers"]].rename(
    columns={"Distance (km)":"Distance_km","Passengers":"Count"}
)
df_after = df[["Distance (km)","Passengers after policy"]].rename(
    columns={"Distance (km)":"Distance_km","Passengers after policy":"Count"}
)
# build lists weighted by count
dist_before = np.repeat(df_before["Distance_km"], df_before["Count"].astype(int))
dist_after  = np.repeat(df_after["Distance_km"], df_after["Count"].astype(int))
dist_range  = dist_before.max() - dist_before.min() if len(dist_before)>0 else 1
bin_size    = dist_range / 50 if dist_range>0 else 1

fig3 = ff.create_distplot(
    [dist_before.tolist(), dist_after.tolist()],
    ["Before","After"],
    bin_size=bin_size,
    show_hist=False,
    show_rug=False
)
fig3.update_layout(
    title="📊 Passenger Distance Distribution: Before vs After",
    xaxis_title="Distance (km)",
    yaxis_title="Density"
)
st.plotly_chart(fig3, use_container_width=True)

# ─────── Kepler country arcs (double height) ───────
req = ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]
if all(c in df.columns for c in req):
    # centroids
    o = df[["Origin Country Name","Origin Lat","Origin Lon"]].rename(
        columns={"Origin Country Name":"Country","Origin Lat":"Lat","Origin Lon":"Lon"}
    )
    d = df[["Destination Country Name","Dest Lat","Dest Lon"]].rename(
        columns={"Destination Country Name":"Country","Dest Lat":"Lat","Dest Lon":"Lon"}
    )
    cents = pd.concat([o,d]).dropna(subset=["Lat","Lon"])\
        .groupby("Country",as_index=False)[["Lat","Lon"]].mean()

    # pair aggregation
    ab = df[["Origin Country Name","Destination Country Name",
             "Passengers","Passengers after policy"]].copy()
    ab["A"] = np.minimum(ab["Origin Country Name"], ab["Destination Country Name"])
    ab["B"] = np.maximum(ab["Origin Country Name"], ab["Destination Country Name"])
    pairs = ab.groupby(["A","B"],as_index=False).sum()
    pairs["Δ (%)"] = (pairs["Passengers after policy"]/pairs["Passengers"] - 1)*100

    # merge coords
    pairs = (pairs
             .merge(cents, left_on="A", right_on="Country").rename(
                columns={"Lat":"A Lat","Lon":"A Lon"}).drop("Country",axis=1)
             .merge(cents, left_on="B", right_on="Country").rename(
                columns={"Lat":"B Lat","Lon":"B Lon"}).drop("Country",axis=1)
    )

    # kepler config
    cfg = {
      "version":"v1","config":{
        "visState":{"layers":[{
          "id":"arc","type":"arc","config":{
            "dataId":"pairs","label":"Δ (%)",
            "columns":{
              "lat0":"A Lat","lng0":"A Lon",
              "lat1":"B Lat","lng1":"B Lon"
            },"isVisible":True,"visConfig":{
              "thickness":3,"opacity":0.8,
              "colorField":{"name":"Δ (%)","type":"real"},
              "colorScale":"quantile",
              "colorRange":{
                "name":"Global Warming","type":"sequential",
                "category":"Uber",
                "colors":["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]
              }
            }
          }
        }],"interactionConfig":{
          "tooltip":{"fieldsToShow":{"pairs":["A","B","Δ (%)"]},"enabled":True}
        }},
        "mapState":{
          "latitude":cents["Lat"].mean(),
          "longitude":cents["Lon"].mean(),
          "zoom":2.2,"pitch":30
        },
        "mapStyle":{}
      }
    }

    kp = KeplerGl(height=1600,data={"pairs":pairs},config=cfg)
    components.html(kp._repr_html_(), height=1620, scrolling=True)

else:
    st.warning("Upload coords with 'Origin Lat', 'Origin Lon', 'Dest Lat', 'Dest Lon' to view Kepler map.")
