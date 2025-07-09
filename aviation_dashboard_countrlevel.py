import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf
from keplergl import KeplerGl
import streamlit.components.v1 as components

# ----------------------
# Model configuration (defaults)
# ----------------------
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND   = 1.4

# ----------------------
# Helper – dummy data
# ----------------------
def load_dummy_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    origins = ["Germany", "France", "United States", "Japan"]
    dests   = ["Spain", "Italy", "United Kingdom", "Canada"]
    rows = []
    for o in origins:
        for d in dests:
            if o == d: continue
            rows.append({
                "Year":                   2022,
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
# UI setup
# ----------------------
st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS – Aviation Simulator")
st.markdown("Simulate air travel between airports and policy impacts.")

# Sidebar uploads
st.sidebar.header("📈 Policy & Data Inputs")
uploaded_file = st.sidebar.file_uploader("Passenger CSV", type=["csv"])
coord_file    = st.sidebar.file_uploader("Coords (.xlsx)", type=["xlsx"])

# ----------------------
# Load & validate data
# ----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Passenger CSV loaded.")
else:
    df = load_dummy_data()
    st.sidebar.info("Using dummy data.")

required_cols = {
    "Origin Country Name","Destination Country Name",
    "Origin Airport","Destination Airport",
    "Distance (km)","Passengers","Avg. Total Fare(USD)"
}
if not required_cols.issubset(df.columns):
    st.error("Passenger CSV missing required columns."); st.stop()

df = df.dropna(subset=required_cols).reset_index(drop=True)

# ----------------------
# Common UI inputs
# ----------------------
origin_all = sorted(df["Origin Country Name"].unique())
dest_all   = sorted(df["Destination Country Name"].unique())

enable_carbon = st.sidebar.checkbox("Enable carbon pricing?")
if enable_carbon:
    ets_price = st.sidebar.slider("Carbon price (EUR/tCO₂)", 0, 400, 100, 5)
    carbon_orig = st.sidebar.multiselect("Carbon: origin countries", origin_all, default=origin_all)
    carbon_dest = st.sidebar.multiselect("Carbon: dest countries",   dest_all,   default=dest_all)
else:
    ets_price = 0.0; carbon_orig = []; carbon_dest = []

enable_tax = st.sidebar.checkbox("Enable air passenger tax?")
if enable_tax:
    air_tax = st.sidebar.slider("Air Passenger Tax (USD)", 0, 100, 10, 1)
    tax_orig = st.sidebar.multiselect("Tax: origin countries", origin_all, default=origin_all)
    tax_dest = st.sidebar.multiselect("Tax: dest countries",   dest_all,   default=dest_all)
else:
    air_tax = 0.0; tax_orig = []; tax_dest = []

st.sidebar.markdown("### Parameters")
pass_through     = st.sidebar.slider("Cost pass-through (%)", 0, 100, 80, 5) / 100
emission_factor  = st.sidebar.slider("Emission factor (kgCO₂/pax-km)", 0.0, 1.0, 0.115, 0.001)

# GDP & elasticity
gdp_global      = st.sidebar.slider("Global GDP growth (%)", -5.0, 8.0, 2.5, 0.1)
price_elast     = st.sidebar.slider("Price elasticity", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1)
gdp_elast       = st.sidebar.slider("GDP elasticity",     0.5,   2.0, GDP_ELASTICITY_DEMAND,   0.1)

st.sidebar.markdown("### Optional: Country-specific GDP growth")
gdp_by_country = {}
with st.sidebar.expander("Customize GDP by origin"):
    for c in origin_all:
        gdp_by_country[c] = st.slider(f"{c} GDP (%)", -5.0, 8.0, gdp_global, 0.1)

# OD‐pair column for regression
df["OD Pair"] = df["Origin Airport"] + "–" + df["Destination Airport"]

# ----------------------
# Policy calculations
# ----------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor

# carbon cost
df["Carbon cost per pax"] = 0.0
if enable_carbon:
    mask = df["Origin Country Name"].isin(carbon_orig) & df["Destination Country Name"].isin(carbon_dest)
    df.loc[mask, "Carbon cost per pax"] = df.loc[mask, "CO2 per pax (kg)"] / 1000 * ets_price * pass_through

# ticket tax
df["Ticket tax per pax"] = 0.0
if enable_tax:
    mask = df["Origin Country Name"].isin(tax_orig) & df["Destination Country Name"].isin(tax_dest)
    df.loc[mask, "Ticket tax per pax"] = air_tax * pass_through

# new fare & %Δ
df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"] + df["Ticket tax per pax"]
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

# elasticity & GDP growth
df["GDP Growth (%)"] = df["Origin Country Name"].map(gdp_by_country).fillna(gdp_global)
df["GDP Factor"]    = (1 + df["GDP Growth (%)"] / 100) ** gdp_elast
df["Fare Factor"]   = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"])**price_elast

df["Passengers after policy"] = df["Passengers"] * df["Fare Factor"] * df["GDP Factor"]
df["Passenger Δ (%)"]         = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# init coords
df["Origin Lat"]=df["Origin Lon"]=df["Dest Lat"]=df["Dest Lon"]=np.nan

# ----------------------
# Load & merge coords
# ----------------------
if coord_file:
    try:
        coords = pd.read_excel(coord_file, engine="openpyxl").drop_duplicates("IATA_Code")
        if {"IATA_Code","DecLat","DecLon"}.issubset(coords.columns):
            cmap = coords.set_index("IATA_Code")[["DecLat","DecLon"]]
            df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
            df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
            df["Origin Lat"]  = df["Origin Code"].map(cmap["DecLat"])
            df["Origin Lon"]  = df["Origin Code"].map(cmap["DecLon"])
            df["Dest Lat"]    = df["Dest Code"].map(cmap["DecLat"])
            df["Dest Lon"]    = df["Dest Code"].map(cmap["DecLon"])
        else:
            st.sidebar.warning("Coords need IATA_Code / DecLat / DecLon")
    except Exception as e:
        st.sidebar.warning(f"Failed to process coords: {e}")

# ----------------------
# Tabs
# ----------------------
tab1, tab2 = st.tabs(["Simulation","Regression"])

with tab1:
    st.subheader("📊 Simulation Results")
    st.dataframe(df[[
        "Origin Airport","Destination Airport","Distance (km)","Passengers",
        "CO2 per pax (kg)","Avg. Total Fare(USD)",
        "Carbon cost per pax","Ticket tax per pax",
        "New Avg Fare","Passenger Δ (%)"
    ]], use_container_width=True)

    # 1) Passenger Δ bar
    or_sum = df.groupby("Origin Country Name",as_index=False).agg(
        Passengers=("Passengers","sum"),
        After=("Passengers after policy","sum")
    )
    or_sum["Δ (%)"] = (or_sum["After"]/or_sum["Passengers"]-1)*100
    fig1 = px.bar(or_sum, "Origin Country Name","Δ (%)",
                  title="Δ Passenger Vol. by Origin")
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Fare Δ bar
    pr_sum = df.groupby("Origin Country Name",as_index=False)["Fare Δ (%)"].mean()
    fig2 = px.bar(pr_sum, "Origin Country Name","Fare Δ (%)",
                  title="Δ Avg Fare by Origin")
    st.plotly_chart(fig2, use_container_width=True)

    # 3) **Smoothed density** curves
    # prepare combined x-grid
    all_dist = df["Distance (km)"].dropna()
    if len(all_dist)>1:
        xs = np.linspace(all_dist.min(), all_dist.max(), 200)
        fig_d = go.Figure()
        for label, subdf in [("Before", df[["Distance (km)","Passengers"]].rename(columns={"Passengers":"Cnt","Distance (km)":"Dist"})),
                              ("After",  df[["Distance (km)","Passengers after policy"]]
                                           .rename(columns={"Passengers after policy":"Cnt","Distance (km)":"Dist"}))]:
            x = subdf["Dist"].to_numpy()
            w = subdf["Cnt"].to_numpy()
            # guard zero‐weight or no variance
            if w.sum() <= 0 or np.nanstd(x) == 0:
                y = np.zeros_like(xs)
            else:
                try:
                    kde = gaussian_kde(x, weights=w)
                    y = kde(xs)
                except Exception:
                    kde = gaussian_kde(x)
                    y = kde(xs)
            fig_d.add_trace(go.Scatter(
                x=xs, y=y, mode="lines", fill="tozeroy", name=label
            ))
        fig_d.update_layout(title="Passenger Distance Density")
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.info("Not enough data for density plot.")

    # 4) Kepler country‐level arcs
    cols = ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]
    if all(c in df.columns for c in cols):
        # centroids
        o = df[["Origin Country Name","Origin Lat","Origin Lon"]].rename(
            {"Origin Country Name":"Country","Origin Lat":"Lat","Origin Lon":"Lon"},axis=1)
        d = df[["Destination Country Name","Dest Lat","Dest Lon"]].rename(
            {"Destination Country Name":"Country","Dest Lat":"Lat","Dest Lon":"Lon"},axis=1)
        cent = pd.concat([o,d],ignore_index=True).dropna(subset=["Lat","Lon"])\
                 .groupby("Country",as_index=False)[["Lat","Lon"]].mean()

        ab = df[["Origin Country Name","Destination Country Name",
                  "Passengers","Passengers after policy"]].copy()
        ab["A"] = np.minimum(ab["Origin Country Name"],ab["Destination Country Name"])
        ab["B"] = np.maximum(ab["Origin Country Name"],ab["Destination Country Name"])
        pa = ab.groupby(["A","B"],as_index=False).sum()
        pa["Δ (%)"] = (pa["Passengers after policy"]/pa["Passengers"]-1)*100

        pa = (pa
              .merge(cent,left_on="A",right_on="Country").rename(
                  {"Lat":"A Lat","Lon":"A Lon"},axis=1).drop("Country",axis=1)
              .merge(cent,left_on="B",right_on="Country").rename(
                  {"Lat":"B Lat","Lon":"B Lon"},axis=1).drop("Country",axis=1))

        cfg = {
          "version":"v1","config":{
            "visState":{"layers":[{
              "id":"arc","type":"arc","config":{
                "dataId":"pa","label":"Δ (%)",
                "columns":{
                  "lat0":"A Lat","lng0":"A Lon",
                  "lat1":"B Lat","lng1":"B Lon"},
                "visConfig":{
                  "thickness":3,"opacity":0.8,
                  "colorField":{"name":"Δ (%)","type":"real"},
                  "colorScale":"quantile",
                  "colorRange":{"colors":["#ffffcc","#41b6c4","#253494"]},
                  "sizeField":"Δ (%)","sizeScale":10
                }
              }
            }]},
            "mapState":{
              "latitude":cent["Lat"].mean(),
              "longitude":cent["Lon"].mean(),
              "zoom":2.2,"pitch":30
            },
            "mapStyle":{}
          }
        }

        km = KeplerGl(height=1600, data={"pa":pa}, config=cfg)
        html = km._repr_html_()
        if isinstance(html, bytes): html = html.decode()
        components.html(html, height=1400, width=1800)
    else:
        st.info("Upload coords to see map.")

with tab2:
    st.subheader("📊 Panel Regression")
    if "Year" not in df.columns:
        st.info("Add a 'Year' column for panel regression.")
    else:
        dep   = st.selectbox("Dependent var",   df.select_dtypes(float).columns)
        indep = st.multiselect("Independent vars", 
                               df.select_dtypes(float).columns.drop(dep, errors="ignore"))
        fe_y  = st.checkbox("Year fixed effects",      value=True)
        fe_u  = st.checkbox("OD‐pair fixed effects",   value=True)
        if st.button("Run regression"):
            formula = f"{dep} ~ " + " + ".join(indep)
            if fe_y: formula += " + C(Year)"
            if fe_u: formula += " + C(`OD Pair`)"
            try:
                res = smf.ols(formula, data=df).fit()
                st.write(res.summary())
            except Exception as e:
                st.error(f"Regression failed: {e}")
