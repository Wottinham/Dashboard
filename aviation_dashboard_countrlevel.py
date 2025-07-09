import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from keplergl import KeplerGl
import streamlit.components.v1 as components
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
            f"{country} GDP growth (%)",
            -5.0, 8.0, global_gdp_growth, 0.1,
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

# New fare & fare Δ
df["New Avg Fare"] = (
    df["Avg. Total Fare(USD)"]
    + df["Carbon cost per pax"]
    + df["Air passenger tax per pax"]
)
df["Fare Δ (%)"] = (
    df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1
) * 100

# Elasticity & GDP adjustments
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

# Initialize coords
for col in ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]:
    df[col] = np.nan

# ----------------------
# Load & merge coordinates
# ----------------------
if coord_file:
    try:
        coords_df = pd.read_excel(coord_file, engine="openpyxl")
        coords_df = coords_df.drop_duplicates(subset=["IATA_Code"])
        if {"IATA_Code","DecLat","DecLon"}.issubset(coords_df.columns):
            coords_map = coords_df.set_index("IATA_Code")[["DecLat","DecLon"]]
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
# Main Tabs
# ----------------------
tab1, tab2 = st.tabs(["Simulation","Regression"])

# ------ Simulation Tab ------
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

    # Barplot: passenger Δ
    origin_summary = df.groupby("Origin Country Name", as_index=False).agg({
        "Passengers":"sum","People after policy":"sum"
    })
    origin_summary["Relative Change (%)"] = (
        origin_summary["Passengers after policy"]/origin_summary["Passengers"] - 1
    )*100
    fig1 = px.bar(
        origin_summary,x="Origin Country Name",y="Relative Change (%)",
        title="📉 Relative Change in Passenger Volume by Origin Country",
        text="Relative Change (%)"
    )
    fig1.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
    st.plotly_chart(fig1,use_container_width=True)

    # Barplot: fare Δ
    price_summary = df.groupby("Origin Country Name", as_index=False)["Fare Δ (%)"].mean()
    fig2 = px.bar(
        price_summary, x="Origin Country Name", y="Fare Δ (%)",
        title="📈 Relative Change in Average Fare by Origin Country",
        text="Fare Δ (%)"
    )
    fig2.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
    st.plotly_chart(fig2,use_container_width=True)

    # Density plot inside Simulation
    st.markdown("### Passenger Distance Density")
    all_dist = df["Distance (km)"].dropna()
    if all_dist.size > 1:
        xs = np.linspace(all_dist.min(),all_dist.max(),200)
        fig_d = go.Figure()
        for label,(dist_col,weight_col) in [
                ("Before",("Distance (km)","Passengers")),
                ("After", ("Distance (km)","Passengers after policy"))
            ]:
            x = df[dist_col].to_numpy()
            w = df[weight_col].to_numpy()
            if w.sum()<=0 or np.nanstd(x)==0:
                y = np.zeros_like(xs)
            else:
                try:
                    kde = gaussian_kde(x,weights=w)
                except Exception:
                    kde = gaussian_kde(x)
                y = kde(xs)
            fig_d.add_trace(go.Scatter(
                x=xs,y=y,mode='lines',fill='tozeroy',name=label
            ))
        fig_d.update_layout(
            xaxis_title="Distance (km)", yaxis_title="Density (scaled by total pax)", height=400
        )
        st.plotly_chart(fig_d,use_container_width=True)
    else:
        st.info("Not enough distance data to draw density.")

    # Kepler map
    req = ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]
    if all(c in df.columns for c in req):
        coords_o = df[["Origin Country Name","Origin Lat","Origin Lon"]]
        coords_o.columns = ["Country","Lat","Lon"]
        coords_d = df[["Destination Country Name","Dest Lat","Dest Lon"]]
        coords_d.columns = ["Country","Lat","Lon"]
        cents = pd.concat([coords_o,coords_d]).dropna().groupby("Country",as_index=False)[["Lat","Lon"]].mean()

        ab = df[["Origin Country Name","Destination Country Name","Passengers","Passengers after policy"]]
        ab["A"] = np.where(ab["Origin Country Name"]<ab["Destination Country Name"],ab["Origin Country Name"],ab["Destination Country Name"])
        ab["B"] = np.where(ab["Origin Country Name"]<ab["Destination Country Name"],ab["Destination Country Name"],ab["Origin Country Name"])
        pair = ab.groupby(["A","B"],as_index=False).sum()
        pair["Traffic Δ (%)"] = (pair["Passengers after policy"]/pair["Passengers"]-1)*100
        pair = pair.merge(cents,left_on="A",right_on="Country").rename(columns={"Lat":"A Lat","Lon":"A Lon"}).drop(columns=["Country"])
        pair = pair.merge(cents,left_on="B",right_on="Country").rename(columns={"Lat":"B Lat","Lon":"B Lon"}).drop(columns=["Country"])

        cfg = {"version":"v1","config":{"visState":{"filters":[],"layers":[{  
                    "id":"arc_layer","type":"arc","config":{  
                        "dataId":"pairs","label":"Traffic Δ (%)","columns":{"lat0":"A Lat","lng0":"A Lon","lat1":"B Lat","lng1":"B Lon"},
                        "isVisible":True,"visConfig":{"thickness":3,"opacity":0.8,
                            "colorField":{"name":"Traffic Δ (%)","type":"real"},
                            "colorScale":"quantile","colorRange":{"name":"Global Warming","type":"sequential","category":"Uber","colors":["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]},
                            "sizeField":"Traffic Δ (%)","sizeScale":10
                        }
                    }}],"interactionConfig":{"tooltip":{"fieldsToShow":{"pairs":["A","B","Traffic Δ (%)"]},"enabled":True}}},
                "mapState":{"latitude":cents["Lat"].mean(),"longitude":cents["Lon"].mean(),"zoom":2.2,"pitch":30},"mapStyle":{}}}
        
        kmap = KeplerGl(height=1600,data={"pairs":pair},config=cfg)
        html = kmap._repr_html_()
        if isinstance(html,bytes): html = html.decode('utf-8')
        components.html(html,width=1800,height=1200)
    else:
        st.warning("Upload coordinates with 'Origin Lat','Origin Lon','Dest Lat','Dest Lon' to see map.")

# ------ Regression Tab ------
with tab2:
    st.subheader("📊 Panel Regression Analysis")
    if 'Year' in df.columns:
        st.write("Detected panel data (Year present). Choose regression settings below.")
        dep = st.selectbox("Dependent variable", [c for c in df.select_dtypes(include=[float,int]).columns])
        indep = st.multiselect("Independent variables", [c for c in df.select_dtypes(include=[float,int]).columns if c!=dep])
        col1, col2 = st.columns(2)
        with col1:
            fe_time = st.checkbox("Include Year fixed effects", value=True)
            fe_unit = st.checkbox("Include OD_Pair fixed effects", value=True)
        # create OD pair identifier
        if 'OD_Pair' not in df.columns and 'Origin Airport' in df.columns:
            df['OD_Pair'] = df['Origin Airport'] + '_' + df['Destination Airport']
        if st.button("Run regression"):
            if not dep or not indep:
                st.error("Select both dependent and at least one independent variable.")
            else:
                formula = f"{dep} ~ {' + '.join(indep)}"
                if fe_time: formula += ' + C(Year)'
                if fe_unit: formula += ' + C(OD_Pair)'
                try:
                    model = smf.ols(formula, data=df).fit()
                    st.write(model.summary())
                except Exception as e:
                    st.error(f"Regression failed: {e}")
    else:
        st.info("Upload panel data with a 'Year' column to enable regression analysis.")
