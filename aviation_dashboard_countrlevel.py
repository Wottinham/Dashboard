import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
uploaded_file = st.sidebar.file_uploader("Upload airport-pair passenger CSV", type=["csv"])
coord_file    = st.sidebar.file_uploader("Upload airport coordinates (.xlsx)", type=["xlsx"])

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
enable_carbon = st.sidebar.checkbox("Enable carbon pricing?")
if enable_carbon:
    ets_price = st.sidebar.slider("Carbon price (EUR / tCO₂)", 0, 400, 100, 5)
    carbon_origin_countries = st.sidebar.multiselect(
        "Origin countries taxed (carbon):", origin_all, default=origin_all
    )
    carbon_dest_countries = st.sidebar.multiselect(
        "Destination countries taxed (carbon):", dest_all, default=dest_all
    )
else:
    ets_price = 0.0
    carbon_origin_countries = []
    carbon_dest_countries = []

# ----------------------
# Air passenger tax policy
# ----------------------
enable_tax = st.sidebar.checkbox("Enable air passenger tax?")
if enable_tax:
    air_passenger_tax = st.sidebar.slider("Air Passenger Tax (USD)", 0, 100, 10, 1)
    tax_origin_countries = st.sidebar.multiselect(
        "Origin countries taxed (tax):", origin_all, default=origin_all
    )
    tax_dest_countries = st.sidebar.multiselect(
        "Destination countries taxed (tax):", dest_all, default=dest_all
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
    "Cost pass-through to fares (%)", 0, 100, 80, 5
) / 100

emission_factor = st.sidebar.slider(
    "Emission factor (kg CO₂ per pax-km)", 0.0, 1.0, 0.115, 0.001
)

# ----------------------
# Economic inputs
# ----------------------
global_gdp_growth = st.sidebar.slider("Global real GDP growth (%)", -5.0, 8.0, 2.5, 0.1)
user_price_elasticity = st.sidebar.slider(
    "Demand price elasticity (negative)", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1
)
user_gdp_elasticity = st.sidebar.slider(
    "Demand GDP elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1
)

st.sidebar.markdown("### Optional: Adjust GDP Growth by Country")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth for Specific Origins"):
    for c in origin_all:
        gdp_growth_by_country[c] = st.slider(
            f"{c} GDP growth (%)", -5.0, 8.0, global_gdp_growth, 0.1
        )

# ----------------------
# Policy calculations
# ----------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor

# Carbon cost
df["Carbon cost per pax"] = 0.0
if enable_carbon:
    m = df["Origin Country Name"].isin(carbon_origin_countries) & \
        df["Destination Country Name"].isin(carbon_dest_countries)
    df.loc[m, "Carbon cost per pax"] = (
        df.loc[m, "CO2 per pax (kg)"] / 1_000 * ets_price * pass_through
    )

# Passenger tax cost
df["Air passenger tax per pax"] = 0.0
if enable_tax:
    m = df["Origin Country Name"].isin(tax_origin_countries) & \
        df["Destination Country Name"].isin(tax_dest_countries)
    df.loc[m, "Air passenger tax per pax"] = air_passenger_tax * pass_through

# New fare & fare Δ
df["New Avg Fare"] = (
    df["Avg. Total Fare(USD)"]
    + df["Carbon cost per pax"]
    + df["Air passenger tax per pax"]
)
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

# Elasticity & GDP impact
factor = (
    (df["New Avg Fare"] / df["Avg. Total Fare(USD)"])
    .replace([np.inf, -np.inf], np.nan) ** user_price_elasticity
)
df["GDP Growth (%)"] = df["Origin Country Name"].map(gdp_growth_by_country).fillna(global_gdp_growth)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"] / 100) ** user_gdp_elasticity

df["Passengers after policy"] = df["Passengers"] * factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# initialize coords
df["Origin Lat"] = np.nan; df["Origin Lon"] = np.nan
df["Dest Lat"]   = np.nan; df["Dest Lon"]   = np.nan

# ----------------------
# Load & merge coordinates
# ----------------------
if coord_file:
    try:
        coords_df = pd.read_excel(coord_file, engine="openpyxl")
        coords_df = coords_df.drop_duplicates(subset=["IATA_Code"])
        if {"IATA_Code","DecLat","DecLon"}.issubset(coords_df.columns):
            cmap = coords_df.set_index("IATA_Code")[["DecLat","DecLon"]]
            df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
            df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
            df["Origin Lat"]  = df["Origin Code"].map(cmap["DecLat"])
            df["Origin Lon"]  = df["Origin Code"].map(cmap["DecLon"])
            df["Dest Lat"]    = df["Dest Code"].map(cmap["DecLat"])
            df["Dest Lon"]    = df["Dest Code"].map(cmap["DecLon"])
        else:
            st.warning("❌ Coordinate file missing IATA_Code / DecLat / DecLon.")
    except ImportError:
        st.warning("❌ Install openpyxl for .xlsx support.")
    except Exception as e:
        st.warning(f"❌ Failed to process coordinates: {e}")

# ----------------------
# Table & bar chart outputs
# ----------------------
st.subheader("📊 Airport-Pair Passenger Results")
st.dataframe(df[[
    "Origin Airport","Destination Airport","Passengers",
    "Distance (km)","CO2 per pax (kg)",
    "Avg. Total Fare(USD)","Carbon cost per pax",
    "Air passenger tax per pax","New Avg Fare","Passenger Δ (%)"
]], use_container_width=True)

# 1) Passenger Δ by origin country
origin_sum = df.groupby("Origin Country Name", as_index=False).agg({
    "Passengers":"sum","Passengers after policy":"sum"
})
origin_sum["Relative Change (%)"] = (
    origin_sum["Passengers after policy"] / origin_sum["Passengers"] - 1
) * 100

fig1 = px.bar(origin_sum,
    x="Origin Country Name", y="Relative Change (%)",
    title="📉 Δ Passengers by Origin Country",
    text="Relative Change (%)"
)
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig1, use_container_width=True)

# metrics
c1,c2 = st.columns(2)
with c1:
    base, new = df["Passengers"].sum(), df["Passengers after policy"].sum()
    st.metric("Total Passengers (M)", f"{new/1e6:.2f}", delta=f"{(new/base-1)*100:+.1f}%")
with c2:
    st.metric("Avg Carbon Cost (€)", f"{df['Carbon cost per pax'].mean():.2f}")

# 2) Fare Δ by origin country
price_sum = df.groupby("Origin Country Name", as_index=False)["Fare Δ (%)"].mean().rename(
    columns={"Fare Δ (%)":"Avg Fare Δ (%)"}
)
fig2 = px.bar(price_sum,
    x="Origin Country Name", y="Avg Fare Δ (%)",
    title="📈 Δ Average Fare by Origin Country",
    text="Avg Fare Δ (%)"
)
fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

# 3) Smoothed density curves
df_b = df[["Distance (km)","Passengers"]].rename(columns={"Distance (km)":"Distance_km","Passengers":"Count"})
df_a = df[["Distance (km)","Passengers after policy"]].rename(columns={"Distance (km)":"Distance_km","Passengers after policy":"Count"})
bins = np.linspace(min(df_b.Distance_km.min(),df_a.Distance_km.min()),
                   max(df_b.Distance_km.max(),df_a.Distance_km.max()), 50)
dens_frames=[]
for label, sub in [("Before",df_b),("After",df_a)]:
    hist,w = np.histogram(sub.Distance_km, bins=bins, weights=sub.Count, density=True)
    centers=0.5*(bins[:-1]+bins[1:])
    dens_frames.append(pd.DataFrame({"Distance (km)":centers,"Density":hist,"Scenario":label}))
dens_df=pd.concat(dens_frames,ignore_index=True)
fig3=px.line(dens_df, x="Distance (km)", y="Density", color="Scenario",
             title="📊 Passenger Distance Density Before vs After")
fig3.update_traces(line_shape="spline")
st.plotly_chart(fig3, use_container_width=True)

# ─────── Kepler country‐level arcs ───────
cols = ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]
if all(c in df.columns for c in cols):
    # centroids
    o = df[["Origin Country Name","Origin Lat","Origin Lon"]].rename(
        columns={"Origin Country Name":"Country","Origin Lat":"Lat","Origin Lon":"Lon"})
    d = df[["Destination Country Name","Dest Lat","Dest Lon"]].rename(
        columns={"Destination Country Name":"Country","Dest Lat":"Lat","Dest Lon":"Lon"})
    cent = pd.concat([o,d],ignore_index=True).dropna(subset=["Lat","Lon"])\
              .groupby("Country",as_index=False)[["Lat","Lon"]].mean()

    # aggregate unordered pairs
    ab=df[["Origin Country Name","Destination Country Name","Passengers","Passengers after policy"]].copy()
    ab["A"]=np.where(ab["Origin Country Name"]<ab["Destination Country Name"],
                     ab["Origin Country Name"], ab["Destination Country Name"])
    ab["B"]=np.where(ab["Origin Country Name"]<ab["Destination Country Name"],
                     ab["Destination Country Name"], ab["Origin Country Name"])
    p=ab.groupby(["A","B"],as_index=False).agg({"Passengers":"sum","Passengers after policy":"sum"})
    p["Traffic Δ (%)"]=(p["Passengers after policy"]/p["Passengers"]-1)*100

    # merge centroids
    p = p.merge(cent, left_on="A", right_on="Country", how="left")\
         .rename(columns={"Lat":"A Lat","Lon":"A Lon"}).drop(columns="Country")\
         .merge(cent, left_on="B", right_on="Country", how="left")\
         .rename(columns={"Lat":"B Lat","Lon":"B Lon"}).drop(columns="Country")

    # kepler config w/ colorField mapped to Traffic Δ (%)
    cfg = {
      "version":"v1","config":{
        "visState":{
          "layers":[{
            "id":"arc","type":"arc","config":{
              "dataId":"pairs","label":"Traffic Δ (%)",
              "columns":{"lat0":"A Lat","lng0":"A Lon","lat1":"B Lat","lng1":"B Lon"},
              "isVisible":True,"visConfig":{
                "thickness":3,"opacity":0.8,
                "colorField":{"name":"Traffic Δ (%)","type":"real"},
                "colorScale":"quantile",
                "colorRange":{
                  "name":"Global Warming","type":"sequential","category":"Uber",
                  "colors":["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]
                },
                "sizeField":"Traffic Δ (%)","sizeScale":10
              }
            }
          }],
          "interactionConfig":{"tooltip":{"fieldsToShow":{"pairs":["A","B","Traffic Δ (%)"]},"enabled":True}}
        },
        "mapState":{
          "latitude":cent["Lat"].mean(),"longitude":cent["Lon"].mean(),"zoom":2.2,"pitch":30
        },
        "mapStyle":{}
      }
    }

    # render full-size 1600×1600
    m = KeplerGl(height=1600, data={"pairs":p}, config=cfg)
    html = m._repr_html_()
    if isinstance(html, bytes): html = html.decode()
    components.html(html, height=1600, width=1600)
else:
    st.warning("Upload coords with 'Origin Lat','Origin Lon','Dest Lat','Dest Lon' to see Kepler map.")
