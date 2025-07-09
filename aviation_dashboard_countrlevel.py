import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from keplergl import KeplerGl
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

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

# Create tabs
tab1, tab2 = st.tabs(["Simulator", "Regression"])

# ----------------------------------------
# Sidebar – first load data so we can define country lists
# ----------------------------------------
st.sidebar.header("📈 Policy & Data Inputs")
uploaded_file = st.sidebar.file_uploader("Upload airport-pair passenger CSV", type=["csv"], key="upload_csv")
coord_file    = st.sidebar.file_uploader("Upload airport coordinates (.xlsx)", type=["xlsx"], key="upload_coords")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Passenger CSV loaded.")
else:
    df = load_dummy_data()
    st.sidebar.info("🛈 No passenger CSV – using **dummy data**.")

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

# Now we can define origin_all and dest_all once
origin_all = sorted(df["Origin Country Name"].unique())
dest_all   = sorted(df["Destination Country Name"].unique())

# ----------------------------------------
# Sidebar – policy inputs
# ----------------------------------------
with st.sidebar:
    # Carbon pricing
    enable_carbon = st.checkbox("Enable carbon pricing?", key="chk_carbon")
    if enable_carbon:
        ets_price = st.slider("Carbon price (EUR / tCO₂)", 0, 400, 100, 5, key="slider_ets")
        carbon_origin_countries = st.multiselect(
            "Origin countries taxed (carbon):", origin_all, default=origin_all, key="mslt_carbon_orig"
        )
        carbon_dest_countries = st.multiselect(
            "Destination countries taxed (carbon):", dest_all, default=dest_all, key="mslt_carbon_dest"
        )
    else:
        ets_price = 0.0
        carbon_origin_countries = []
        carbon_dest_countries = []

    # Passenger tax
    enable_tax = st.checkbox("Enable air passenger tax?", key="chk_tax")
    if enable_tax:
        air_passenger_tax = st.slider("Air Passenger Tax (USD)", 0, 100, 10, 1, key="slider_tax")
        tax_origin_countries = st.multiselect(
            "Origin countries taxed (tax):", origin_all, default=origin_all, key="mslt_tax_orig"
        )
        tax_dest_countries = st.multiselect(
            "Destination countries taxed (tax):", dest_all, default=dest_all, key="mslt_tax_dest"
        )
    else:
        air_passenger_tax = 0.0
        tax_origin_countries = []
        tax_dest_countries = []

    st.markdown("### Parameters")
    pass_through = st.slider(
        "Cost pass-through to fares (%)", 0, 100, 80, 5,
        help="Share of carbon cost and ticket tax airlines embed in ticket prices.",
        key="slider_pass_through"
    ) / 100

    emission_factor = st.slider(
        "Emission factor (kg CO₂ per pax-km)", 0.0, 1.0, 0.115, 0.001,
        help="kg of CO₂ emitted per passenger-km flown.",
        key="slider_emission_factor"
    )

    global_gdp_growth = st.slider("Global real GDP growth (%)", -5.0, 8.0, 2.5, 0.1, key="slider_gdp_global")
    user_price_elast  = st.slider(
        "Demand price elasticity (negative)", -2.0, -0.1,
        PRICE_ELASTICITY_DEMAND, 0.1, key="slider_price_elast"
    )
    user_gdp_elast    = st.slider(
        "Demand GDP elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1,
        key="slider_gdp_elast"
    )

    st.markdown("### Optional: Adjust GDP Growth by Country")
    gdp_growth_by_country = {}
    with st.expander("Customize GDP Growth for Specific Origins"):
        for country in origin_all:
            gdp_growth_by_country[country] = st.slider(
                f"{country} GDP growth (%)", -5.0, 8.0, global_gdp_growth, 0.1,
                key=f"gdp_{country}"
            )

# ----------------------------------------
# Shared policy calculations
# ----------------------------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor

# Carbon cost
df["Carbon cost per pax"] = 0.0
if enable_carbon:
    mask_c = (
        df["Origin Country Name"].isin(carbon_origin_countries) &
        df["Destination Country Name"].isin(carbon_dest_countries)
    )
    df.loc[mask_c, "Carbon cost per pax"] = (df.loc[mask_c, "CO2 per pax (kg)"] / 1e3) * ets_price * pass_through

# Passenger tax
df["Air passenger tax per pax"] = 0.0
if enable_tax:
    mask_t = (
        df["Origin Country Name"].isin(tax_origin_countries) &
        df["Destination Country Name"].isin(tax_dest_countries)
    )
    df.loc[mask_t, "Air passenger tax per pax"] = air_passenger_tax * pass_through

# New fare & % change
df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"] + df["Air passenger tax per pax"]
df["Fare Δ (%)"]    = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

# Elasticity & GDP factor
fare_factor = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"]).replace([np.inf, -np.inf], np.nan) ** user_price_elast
df["GDP Growth (%)"]    = df["Origin Country Name"].map(gdp_growth_by_country).fillna(global_gdp_growth)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"]/100) ** user_gdp_elast
df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"]         = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# Initialize coords
df["Origin Lat"] = np.nan; df["Origin Lon"] = np.nan
df["Dest Lat"]   = np.nan; df["Dest Lon"]   = np.nan

# ----------------------------------------
# Load & merge coordinates (if provided)
# ----------------------------------------
if coord_file:
    try:
        coords_df = pd.read_excel(coord_file, engine="openpyxl").drop_duplicates(subset=["IATA_Code"])
        if {"IATA_Code","DecLat","DecLon"}.issubset(coords_df.columns):
            cmap = coords_df.set_index("IATA_Code")[["DecLat","DecLon"]]
            df["Origin Code"] = df["Origin Airport"].str.split("-",1).str[0]
            df["Dest Code"]   = df["Destination Airport"].str.split("-",1).str[0]
            df["Origin Lat"]  = df["Origin Code"].map(cmap["DecLat"])
            df["Origin Lon"]  = df["Origin Code"].map(cmap["DecLon"])
            df["Dest Lat"]    = df["Dest Code"].map(cmap["DecLat"])
            df["Dest Lon"]    = df["Dest Code"].map(cmap["DecLon"])
        else:
            st.sidebar.warning("❌ Coordinate file missing IATA_Code/DecLat/DecLon.")
    except ImportError:
        st.sidebar.warning("❌ Install openpyxl: pip install openpyxl")
    except Exception as e:
        st.sidebar.warning(f"❌ Failed to process coords: {e}")

# ----------------------------------------
# Tab 1: Simulator (cross-sectional)
# ----------------------------------------
with tab1:
    st.subheader("📊 Airport-Pair Passenger Results")
    st.dataframe(
        df[[
            "Origin Airport","Destination Airport","Passengers",
            "Distance (km)","CO2 per pax (kg)",
            "Avg. Total Fare(USD)","Carbon cost per pax",
            "Air passenger tax per pax","New Avg Fare",
            "Passenger Δ (%)"
        ]], use_container_width=True
    )

    # 1) Δ Passengers by origin country
    origin_summary = df.groupby("Origin Country Name", as_index=False).agg({
        "Passengers":              "sum",
        "Passengers after policy": "sum"
    })
    origin_summary["Relative Change (%)"] = (
        origin_summary["Passengers after policy"] / origin_summary["Passengers"] - 1
    ) * 100

    fig1 = px.bar(
        origin_summary, x="Origin Country Name", y="Relative Change (%)",
        title="📉 Relative Change in Passenger Volume by Origin Country",
        text="Relative Change (%)", labels={"Relative Change (%)":"Δ Passengers (%)"}
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    # KPIs
    col1, col2 = st.columns(2)
    with col1:
        base = df["Passengers"].sum()
        new  = df["Passengers after policy"].sum()
        st.metric("Total Passengers (M)", f"{new/1e6:,.2f}", delta=f"{(new/base-1)*100:+.1f}%")
    with col2:
        st.metric("Avg. Carbon Cost (€)", f"{df['Carbon cost per pax'].mean():.2f}")

    # 2) Δ Fares by origin country
    origin_price = df.groupby("Origin Country Name", as_index=False).agg({"Fare Δ (%)":"mean"}).rename(
        columns={"Fare Δ (%)":"Avg Fare Δ (%)"}
    )
    fig2 = px.bar(
        origin_price, x="Origin Country Name", y="Avg Fare Δ (%)",
        title="📈 Relative Change in Average Fare by Origin Country",
        text="Avg Fare Δ (%)", labels={"Avg Fare Δ (%)":"Δ Fare (%)"}
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # 3) Smoothed density curves
    st.subheader("📊 Passenger Distance Density Curves")
    fig, ax = plt.subplots(figsize=(7,7))
    df["Distance (km)"].plot.density(ax=ax, linewidth=4, label="Before")
    df["Distance (km)"].plot.density(ax=ax, linewidth=4, label="After")
    ax.set_xlabel("Distance (km)")
    ax.legend()
    st.pyplot(fig)

    # 4) Kepler map (country-level arcs)
    required = ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]
    if all(c in df.columns for c in required):
        co = df[["Origin Country Name","Origin Lat","Origin Lon"]].rename(
            columns={"Origin Country Name":"Country","Origin Lat":"Lat","Origin Lon":"Lon"}
        )
        cd = df[["Destination Country Name","Dest Lat","Dest Lon"]].rename(
            columns={"Destination Country Name":"Country","Dest Lat":"Lat","Dest Lon":"Lon"}
        )
        cents = pd.concat([co,cd],ignore_index=True).dropna(subset=["Lat","Lon"])
        cents = cents.groupby("Country",as_index=False)[["Lat","Lon"]].mean()

        ab = df[[
            "Origin Country Name","Destination Country Name",
            "Passengers","Passengers after policy"
        ]].copy()
        ab["A"] = np.where(
            ab["Origin Country Name"] < ab["Destination Country Name"],
            ab["Origin Country Name"], ab["Destination Country Name"]
        )
        ab["B"] = np.where(
            ab["Origin Country Name"] < ab["Destination Country Name"],
            ab["Destination Country Name"], ab["Origin Country Name"]
        )
        pa = ab.groupby(["A","B"],as_index=False).agg(
            {"Passengers":"sum","Passengers after policy":"sum"}
        )
        pa["Traffic Δ (%)"] = (pa["Passengers after policy"]/pa["Passengers"] - 1)*100

        pa = (
            pa
            .merge(cents,left_on="A",right_on="Country",how="left")
            .rename(columns={"Lat":"A Lat","Lon":"A Lon"})
            .drop(columns=["Country"])
            .merge(cents,left_on="B",right_on="Country",how="left")
            .rename(columns={"Lat":"B Lat","Lon":"B Lon"})
            .drop(columns=["Country"])
        )

        cfg = {
          "version":"v1","config":{
            "visState":{
              "filters":[],
              "layers":[{
                "id":"arc","type":"arc","config":{
                  "dataId":"pairs","label":"Traffic Δ (%)",
                  "columns":{
                    "lat0":"A Lat","lng0":"A Lon",
                    "lat1":"B Lat","lng1":"B Lon"
                  },
                  "visConfig":{
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
              }]
            },
            "mapState":{
              "latitude":cents["Lat"].mean(),
              "longitude":cents["Lon"].mean(),
              "zoom":2.2,"pitch":30
            },
            "mapStyle":{}
          }
        }

        km = KeplerGl(height=1600, data={"pairs":pa}, config=cfg)
        raw = km._repr_html_()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        components.html(raw, height=1200, width=1800, scrolling=True)
    else:
        st.warning("Upload coords with Origin/Dest Lat/Lon to see Kepler map.")

# ----------------------------------------
# Tab 2: Regression
# ----------------------------------------
with tab2:
    st.subheader("📊 Regression Analysis")
    is_panel = "Year" in df.columns
    st.info("Panel data detected." if is_panel else "Cross-sectional data.")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    dep_var  = st.selectbox("Dependent variable", num_cols, key="dep")
    indeps   = st.multiselect(
        "Independent variables", [c for c in num_cols if c != dep_var], key="indep"
    )
    fe_choices = ["Origin Country Name", "Destination Country Name"]
    if is_panel:
        fe_choices.append("Year")
    fe_sel = st.multiselect("Fixed effects", fe_choices, key="fe")

    if st.button("Run regression", key="run_reg"):
        if not indeps:
            st.error("Select at least one independent variable.")
        else:
            formula = f"`{dep_var}` ~ " + " + ".join(f"`{v}`" for v in indeps)
            for fe in fe_sel:
                formula += " + C(`" + fe + "`)"
            with st.spinner("Running regression..."):
                try:
                    mod = smf.ols(formula, data=df).fit()
                    st.text(f"Formula: {formula}")
                    st.text(mod.summary().as_text())
                except Exception as e:
                    st.error(f"Regression failed: {e}")
