import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from keplergl import KeplerGl
import streamlit.components.v1 as components
from scipy.stats import gaussian_kde

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
coord_file    = st.sidebar.file_uploader("Upload airport coords (.xlsx)", type=["xlsx"])

# ----------------------
# Load passenger data
# ----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Passenger CSV loaded.")
else:
    df = load_dummy_data()
    st.sidebar.info("No passenger CSV – using dummy data.")

# Validate passenger columns
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
# Detect panel data
# ----------------------
panel_data = "Year" in df.columns

# ----------------------
# Simulation inputs (always shown)
# ----------------------
enable_carbon = st.sidebar.checkbox("Enable carbon pricing")
if enable_carbon:
    ets_price = st.sidebar.slider("Carbon price (EUR/tCO₂)", 0, 400, 100, 5)
    carbon_origin = st.sidebar.multiselect("Carbon taxed: Origin countries", origin_all, default=origin_all)
    carbon_dest   = st.sidebar.multiselect("Carbon taxed: Destination countries", dest_all, default=dest_all)
else:
    ets_price = 0.0
    carbon_origin = []
    carbon_dest = []

enable_tax = st.sidebar.checkbox("Enable air passenger tax")
if enable_tax:
    air_pass_tax = st.sidebar.slider("Air Passenger Tax (USD)", 0, 100, 10, 1)
    tax_origin   = st.sidebar.multiselect("Taxed: Origin countries", origin_all, default=origin_all)
    tax_dest     = st.sidebar.multiselect("Taxed: Destination countries", dest_all, default=dest_all)
else:
    air_pass_tax = 0.0
    tax_origin = []
    tax_dest = []

st.sidebar.markdown("### Parameters")
pass_through     = st.sidebar.slider("Cost pass-through (%)", 0, 100, 80, 5) / 100
emission_factor  = st.sidebar.slider("Emission factor (kg CO₂/pax-km)", 0.0, 1.0, 0.115, 0.001)

global_gdp   = st.sidebar.slider("Global GDP growth (%)", -5.0, 8.0, 2.5, 0.1)
price_elast  = st.sidebar.slider("Price elasticity (neg)", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1)
gdp_elast    = st.sidebar.slider("GDP elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1)

st.sidebar.markdown("### Optional: GDP by origin")
gdp_by_country = {}
with st.sidebar.expander("Customize GDP growth"):
    for c in origin_all:
        gdp_by_country[c] = st.slider(
            f"{c} GDP growth (%)", -5.0, 8.0, global_gdp, 0.1, key=f"gdp_{c}"
        )

# ----------------------
# Apply policies to data
# ----------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor
df["Carbon cost per pax"]   = 0.0
if enable_carbon:
    mask_c = df["Origin Country Name"].isin(carbon_origin) & df["Destination Country Name"].isin(carbon_dest)
    df.loc[mask_c, "Carbon cost per pax"] = df.loc[mask_c, "CO2 per pax (kg)"] / 1000 * ets_price * pass_through

df["Air passenger tax per pax"] = 0.0
if enable_tax:
    mask_t = df["Origin Country Name"].isin(tax_origin) & df["Destination Country Name"].isin(tax_dest)
    df.loc[mask_t, "Air passenger tax per pax"] = air_pass_tax * pass_through

df["New Avg Fare"] = (
    df["Avg. Total Fare(USD)"]
    + df["Carbon cost per pax"]
    + df["Air passenger tax per pax"]
)
df["Fare Δ (%)"] = (
    df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1
) * 100

fare_factor = (
    (df["New Avg Fare"] / df["Avg. Total Fare(USD)"])
    .replace([np.inf, -np.inf], np.nan) ** price_elast
)
df["GDP Growth (%)"]    = df["Origin Country Name"].map(gdp_by_country).fillna(global_gdp)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"]/100) ** gdp_elast

df["Passengers after policy"] = (
    df["Passengers"] * fare_factor * df["GDP Growth Factor"]
)
df["Passenger Δ (%)"] = (
    df["Passengers after policy"] / df["Passengers"] - 1
) * 100

# Initialize coords
df["Origin Lat"] = np.nan; df["Origin Lon"] = np.nan
df["Dest Lat"]   = np.nan; df["Dest Lon"]   = np.nan

# ----------------------
# Load & merge coordinates
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
            st.sidebar.warning("Coords file missing IATA_Code/DecLat/DecLon")
    except ImportError:
        st.sidebar.warning("Install openpyxl to read .xlsx")
    except Exception as e:
        st.sidebar.warning(f"Failed coords processing: {e}")

# ----------------------
# Top-level tabs: Simulation vs Regression
# ----------------------
tab_sim, tab_reg = st.tabs(["Simulation", "Regression"])

# ---- Simulation tab ----
with tab_sim:
    sub1, sub2 = st.tabs(["Direct effects", "Catalytic effects"])

    # Direct effects with nested tabs
    with sub1:
        tab_me, tab_supply = st.tabs(["Market Equilibrium", "Supply"])

        # Market Equilibrium: current direct-effects analysis
        with tab_me:
            # ... (unchanged code for Market Equilibrium) ...
            pass

        # Supply: HHI analysis on Operating Airline Capacity adjusted by passenger change
        with tab_supply:
            st.subheader("📦 Supply-side HHI & Capacity Share Analysis")
            supply_file = st.file_uploader(
                "Upload supply CSV",
                type=["csv"],
                help="Columns: Origin Airport, Destination Airport, Operating Airline, Operating Airline   Capacity",
                key="supply"
            )
            if supply_file:
                supply_df = pd.read_csv(supply_file)
                req_sup = {
                    "Origin Airport","Destination Airport",
                    "Operating Airline","Operating Airline   Capacity"
                }
                if not req_sup.issubset(supply_df.columns):
                    st.error("Supply CSV missing required columns.")
                else:
                    # Merge country names from passenger df
                    orig_map = df[["Origin Airport","Origin Country Name"]].drop_duplicates()
                    dest_map = df[["Destination Airport","Destination Country Name"]].drop_duplicates()
                    supply_df = supply_df.merge(orig_map, on="Origin Airport", how="left")
                    supply_df = supply_df.merge(dest_map, on="Destination Airport", how="left")

                    # Merge passenger change per airport-pair
                    pass_change = df[[
                        "Origin Airport","Destination Airport","Passenger Δ (%)"
                    ]]
                    supply_df = supply_df.merge(
                        pass_change,
                        on=["Origin Airport","Destination Airport"],
                        how="left"
                    )
                    supply_df["Passenger Δ (%)"] = supply_df["Passenger Δ (%)"].fillna(0)

                    # Adjust capacity by passenger change
                    supply_df["Adj Capacity"] = (
                        supply_df["Operating Airline   Capacity"]
                        * (1 + supply_df["Passenger Δ (%)"] / 100)
                    )

                    # Compute HHI per airport-pair using adjusted capacity
                    def compute_hhi(g):
                        caps = g["Adj Capacity"].astype(float)
                        shares = caps / caps.sum()
                        return (shares**2).sum() * 10000

                    hhi = (
                        supply_df
                        .groupby(
                            ["Origin Country Name","Origin Airport","Destination Airport"],
                            as_index=False
                        )
                        .apply(lambda g: pd.Series({"HHI": compute_hhi(g)}))
                        .reset_index(drop=True)
                    )

                    # Box plot of HHI by Origin Country
                    fig_hhi = px.box(
                        hhi,
                        x="Origin Country Name",
                        y="HHI",
                        title="HHI per Airport Pair by Origin Country (Adjusted Capacity)",
                        labels={"HHI":"HHI Index"}
                    )
                    st.plotly_chart(fig_hhi, use_container_width=True)

                    # Pie charts: capacity share by airline, per Origin Country
                    for country in supply_df["Origin Country Name"].unique():
                        pie_df = (
                            supply_df[supply_df["Origin Country Name"] == country]
                            .groupby("Operating Airline", as_index=False)
                            .agg({"Adj Capacity":"sum"})
                        )
                        fig_pie = px.pie(
                            pie_df,
                            names="Operating Airline",
                            values="Adj Capacity",
                            title=f"{country}: Operating Airline Capacity Share"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Upload a supply CSV to see HHI & capacity share analysis.")

    # Catalytic effects
    with sub2:
        # ... (unchanged Catalytic effects code) ...
        pass

# ---- Regression tab ----
with tab_reg:
    st.subheader("📊 Panel Regression Analysis")
    if panel_data:
        st.write("Detected 'Year' column – ready for regression.")
        # Regression UI and output go here...
    else:
        st.info("Upload panel data with a 'Year' column to enable regression mode.")
