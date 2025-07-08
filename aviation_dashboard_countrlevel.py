import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
from keplergl import KeplerGl
import streamlit.components.v1 as components

# ----------------------
# Model configuration (defaults)
# ----------------------
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND   = 1.4

# ----------------------
# Helper – dummy cross‐sectional data
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
st.set_page_config(page_title="Airport‐Pair Simulator", layout="wide")
st.title("✈️ JETPAS – Aviation Simulator & Panel Regression")
st.markdown("Upload either a cross-sectional OD file **or** panel OD data with a `Year` column.")

# ----------------------
# Upload CSV
# ----------------------
uploaded_file = st.sidebar.file_uploader(
    "1) Upload your OD CSV", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded CSV with columns: " + ", ".join(df.columns))
else:
    df = load_dummy_data()
    st.sidebar.info("Using dummy cross‐sectional data.")

# ----------------------
# Detect panel vs cross‐section
# ----------------------
is_panel = "Year" in df.columns

if is_panel:
    # ----------------------
    # PANEL REGRESSION MODE
    # ----------------------
    st.header("📊 Panel Regression Analysis")
    st.warning("Detected `Year` column – running panel OLS with fixed effects.")

    # Create a unit‐ID for each airport‐pair if not already present
    if "OD Pair" not in df.columns:
        df["OD Pair"] = df["Origin Airport"] + " ↔ " + df["Destination Airport"]

    # Sidebar controls for regression
    st.sidebar.header("Regression Settings")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    dep_var      = st.sidebar.selectbox("Dependent variable", numeric_cols)
    indep_vars   = st.sidebar.multiselect("Independent variables", numeric_cols, default=[c for c in numeric_cols if c != dep_var][:2])
    unit_fe      = st.sidebar.selectbox("Unit fixed‐effect (panel ID)", ["OD Pair"], index=0)
    time_fe      = st.sidebar.selectbox("Time fixed‐effect", ["Year"], index=0)

    if st.sidebar.button("Run Regression"):
        if not dep_var or not indep_vars:
            st.error("Please select both a dependent and at least one independent.")
        else:
            # build formula: y ~ x1 + x2 + C(unit_fe) + C(time_fe)
            rhs = " + ".join(indep_vars + [f"C({unit_fe})", f"C({time_fe})"])
            formula = f"{dep_var} ~ {rhs}"
            try:
                model = smf.ols(formula=formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[unit_fe]})
                st.subheader("Regression Results")
                st.text(model.summary().as_text())
            except Exception as e:
                st.error(f"Regression failed: {e}")

else:
    # ----------------------
    # CROSS-SECTIONAL SIMULATOR MODE
    # ----------------------
    st.header("📈 Cross‐Sectional Simulator")

    # Sidebar – coordinate upload
    coord_file = st.sidebar.file_uploader(
        "2) (Optional) Upload airport coords (.xlsx)", type=["xlsx"]
    )

    # Validate core columns
    required = {
        "Origin Country Name", "Destination Country Name",
        "Origin Airport",        "Destination Airport",
        "Distance (km)",         "Passengers",
        "Avg. Total Fare(USD)"
    }
    if not required.issubset(df.columns):
        st.error("CSV missing required OD columns.")
        st.stop()

    df = df.dropna(subset=required).reset_index(drop=True)

    # prepare country lists
    origin_all = sorted(df["Origin Country Name"].unique())
    dest_all   = sorted(df["Destination Country Name"].unique())

    # Carbon pricing
    st.sidebar.markdown("### Carbon Pricing")
    enable_carbon = st.sidebar.checkbox("Enable carbon pricing?")
    if enable_carbon:
        ets_price = st.sidebar.slider("Price (EUR/tCO₂)", 0, 400, 100, 5)
        c_orig = st.sidebar.multiselect("Carbon: origin countries", origin_all, default=origin_all)
        c_dest = st.sidebar.multiselect("Carbon: destination countries", dest_all, default=dest_all)
    else:
        ets_price, c_orig, c_dest = 0.0, [], []

    # Air passenger tax
    st.sidebar.markdown("### Air Passenger Tax")
    enable_tax = st.sidebar.checkbox("Enable passenger tax?")
    if enable_tax:
        tax_amt = st.sidebar.slider("Tax (USD)", 0, 100, 10, 1)
        t_orig = st.sidebar.multiselect("Tax: origin countries", origin_all, default=origin_all)
        t_dest = st.sidebar.multiselect("Tax: destination countries", dest_all, default=dest_all)
    else:
        tax_amt, t_orig, t_dest = 0.0, [], []

    # Parameters
    st.sidebar.markdown("### Parameters")
    pass_through    = st.sidebar.slider("Cost pass‐through (%)", 0, 100, 80, 5) / 100
    emission_factor = st.sidebar.slider("Emission factor (kg CO₂/pax‐km)", 0.0, 1.0, 0.115, 0.001)

    # Economic inputs
    st.sidebar.markdown("### Economic Inputs")
    global_gdp = st.sidebar.slider("Global GDP growth (%)", -5.0, 8.0, 2.5, 0.1)
    price_elast= st.sidebar.slider("Price elasticity", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1)
    gdp_elast  = st.sidebar.slider("GDP elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1)

    st.sidebar.markdown("### Optional: Country‐specific GDP")
    gdp_by_cty = {}
    with st.sidebar.expander("Adjust GDP growth by origin"):
        for c in origin_all:
            gdp_by_cty[c] = st.slider(f"{c} GDP (%)", -5.0, 8.0, global_gdp, 0.1)

    # --- simulate policy effects ---
    df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor
    # carbon cost
    df["Carbon cost per pax"] = 0.0
    if enable_carbon:
        m = df["Origin Country Name"].isin(c_orig) & df["Destination Country Name"].isin(c_dest)
        df.loc[m, "Carbon cost per pax"] = df.loc[m, "CO2 per pax (kg)"] / 1000 * ets_price * pass_through
    # tax cost
    df["Air passenger tax per pax"] = 0.0
    if enable_tax:
        m = df["Origin Country Name"].isin(t_orig) & df["Destination Country Name"].isin(t_dest)
        df.loc[m, "Air passenger tax per pax"] = tax_amt * pass_through

    # new fares & deltas
    df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"] + df["Air passenger tax per pax"]
    df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

    # elasticity & GDP
    ff = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"]).replace([np.inf,-np.inf], np.nan) ** price_elast
    df["GDP Growth (%)"] = df["Origin Country Name"].map(gdp_by_cty).fillna(global_gdp)
    df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"] / 100) ** gdp_elast
    df["Passengers after policy"] = df["Passengers"] * ff * df["GDP Growth Factor"]
    df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

    # initialize coords
    df["Origin Lat"] = np.nan; df["Origin Lon"] = np.nan
    df["Dest Lat"]   = np.nan; df["Dest Lon"]   = np.nan

    # optionally merge coords
    if coord_file:
        try:
            coords_df = pd.read_excel(coord_file, engine="openpyxl").drop_duplicates("IATA_Code")
            if {"IATA_Code","DecLat","DecLon"}.issubset(coords_df.columns):
                cmap = coords_df.set_index("IATA_Code")[["DecLat","DecLon"]]
                df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
                df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
                df["Origin Lat"]  = df["Origin Code"].map(cmap["DecLat"])
                df["Origin Lon"]  = df["Origin Code"].map(cmap["DecLon"])
                df["Dest Lat"]    = df["Dest Code"].map(cmap["DecLat"])
                df["Dest Lon"]    = df["Dest Code"].map(cmap["DecLon"])
            else:
                st.sidebar.warning("Coords must have IATA_Code, DecLat, DecLon")
        except ImportError:
            st.sidebar.warning("Install openpyxl for XLSX support")
        except Exception as e:
            st.sidebar.warning(f"Failed coords: {e}")

    # --- outputs ---
    st.subheader("📊 Airport-Pair Results")
    st.dataframe(df[[
      "Origin Airport","Destination Airport","Passengers","Distance (km)",
      "CO2 per pax (kg)","Avg. Total Fare(USD)","Carbon cost per pax",
      "Air passenger tax per pax","New Avg Fare","Passenger Δ (%)"
    ]], use_container_width=True)

    # passenger Δ by origin country
    agg1 = df.groupby("Origin Country Name", as_index=False).agg({
        "Passengers":"sum","Passengers after policy":"sum"
    })
    agg1["Δ Passengers (%)"] = (agg1["Passengers after policy"]/agg1["Passengers"]-1)*100
    fig1 = px.bar(agg1, x="Origin Country Name", y="Δ Passengers (%)", text="Δ Passengers (%)",
                  title="Δ Passenger Volume by Origin")
    fig1.update_traces(texttemplate="%{text:.1f}%")
    st.plotly_chart(fig1, use_container_width=True)

    # fare Δ by origin
    agg2 = df.groupby("Origin Country Name", as_index=False)["Fare Δ (%)"].mean().rename(
        columns={"Fare Δ (%)":"Avg Δ Fare (%)"}
    )
    fig2 = px.bar(agg2, x="Origin Country Name", y="Avg Δ Fare (%)", text="Avg Δ Fare (%)",
                  title="Δ Average Fare by Origin")
    fig2.update_traces(texttemplate="%{text:.1f}%")
    st.plotly_chart(fig2, use_container_width=True)

    # distance‐density
    before = df[["Distance (km)","Passengers"]].rename(columns={"Distance (km)":"Dist","Passengers":"Cnt"})
    after  = df[["Distance (km)","Passengers after policy"]].rename(columns={"Distance (km)":"Dist","Passengers after policy":"Cnt"})
    bins   = np.linspace(min(before.Dist.min(),after.Dist.min()),
                         max(before.Dist.max(),after.Dist.max()),50)
    dens_dfs=[]
    for label, subset in [("Before",before),("After",after)]:
        h,w= np.histogram(subset.Dist, bins=bins, weights=subset.Cnt, density=True)
        centers=0.5*(bins[:-1]+bins[1:])
        dens_dfs.append(pd.DataFrame({"Dist":centers,"Density":h,"Scenario":label}))
    dens_df=pd.concat(dens_dfs,ignore_index=True)
    fig3=px.line(dens_df, x="Dist", y="Density", color="Scenario",
                 title="Passenger Distance Density: Before vs After")
    fig3.update_traces(line_shape="spline")
    st.plotly_chart(fig3, use_container_width=True)

    # Kepler arcs (country level)
    cols = ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]
    if all(c in df.columns for c in cols):
        o = df[["Origin Country Name","Origin Lat","Origin Lon"]].rename(
            columns={"Origin Country Name":"Country","Origin Lat":"Lat","Origin Lon":"Lon"})
        d = df[["Destination Country Name","Dest Lat","Dest Lon"]].rename(
            columns={"Destination Country Name":"Country","Dest Lat":"Lat","Dest Lon":"Lon"})
        cent = pd.concat([o,d],ignore_index=True).dropna(subset=["Lat","Lon"])\
                 .groupby("Country",as_index=False)[["Lat","Lon"]].mean()
        ab = df[["Origin Country Name","Destination Country Name","Passengers","Passengers after policy"]].copy()
        ab["A"]=np.where(ab["Origin Country Name"]<ab["Destination Country Name"],
                         ab["Origin Country Name"],ab["Destination Country Name"])
        ab["B"]=np.where(ab["Origin Country Name"]<ab["Destination Country Name"],
                         ab["Destination Country Name"],ab["Origin Country Name"])
        p = ab.groupby(["A","B"],as_index=False).agg({"Passengers":"sum","Passengers after policy":"sum"})
        p["Traffic Δ (%)"] = (p["Passengers after policy"]/p["Passengers"]-1)*100
        p = p.merge(cent,left_on="A",right_on="Country")\
             .rename(columns={"Lat":"A Lat","Lon":"A Lon"}).drop(columns="Country")\
             .merge(cent,left_on="B",right_on="Country")\
             .rename(columns={"Lat":"B Lat","Lon":"B Lon"}).drop(columns="Country")
        cfg={
          "version":"v1","config":{
            "visState":{
              "layers":[{
                "id":"arc","type":"arc","config":{
                  "dataId":"pairs","label":"Traffic Δ (%)",
                  "columns":{"lat0":"A Lat","lng0":"A Lon","lat1":"B Lat","lng1":"B Lon"},
                  "visConfig":{
                    "thickness":3,"opacity":0.8,
                    "colorField":{"name":"Traffic Δ (%)","type":"real"},
                    "colorScale":"quantile",
                    "colorRange":{"name":"Global Warming","type":"sequential","category":"Uber",
                                  "colors":["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]},
                    "sizeField":"Traffic Δ (%)","sizeScale":10
                  }
                }
              }],
              "interactionConfig":{"tooltip":{"fieldsToShow":{"pairs":["A","B","Traffic Δ (%)"]},"enabled":True}}
            },
            "mapState":{
              "latitude":cent["Lat"].mean(),"longitude":cent["Lon"].mean(),
              "zoom":2.2,"pitch":30
            },
            "mapStyle":{}
          }
        }
        m = KeplerGl(height=1600, data={"pairs":p}, config=cfg)
        raw = m._repr_html_()
        if isinstance(raw, bytes): raw = raw.decode()
        components.html(raw, height=1200, width=1800)
    else:
        st.sidebar.info("Upload coords (.xlsx) with IATA_Code/DecLat/DecLon to enable map.")
