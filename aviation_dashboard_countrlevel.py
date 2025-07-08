import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
from keplergl import KeplerGl
import streamlit.components.v1 as components

# ----------------------
# Model configuration
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
# Streamlit setup
# ----------------------
st.set_page_config(page_title="JETPAS Simulator", layout="wide")
st.title("✈️ JETPAS – Aviation Simulator & Panel Regression")
st.markdown("Upload a cross-section OD file or panel OD data (with a `Year` column).")

# ----------------------
# Upload data
# ----------------------
uploaded = st.sidebar.file_uploader("1) Upload your OD CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("CSV loaded: " + ", ".join(df.columns))
else:
    df = load_dummy_data()
    st.sidebar.info("Using dummy cross-section data.")

# ----------------------
# Panel vs cross-section?
# ----------------------
is_panel = "Year" in df.columns

if is_panel:
    # ─────────────────────────────────────────────────────────────────
    # PANEL REGRESSION MODE
    # ─────────────────────────────────────────────────────────────────
    st.header("📊 Panel Regression Analysis")
    st.warning("Detected `Year` → panel OLS with selectable fixed effects.")

    # create an OD Pair identifier if missing
    if "OD Pair" not in df.columns:
        df["OD Pair"] = df["Origin Airport"] + " ↔ " + df["Destination Airport"]

    # regression controls in sidebar
    st.sidebar.header("Regression Settings")
    numeric = df.select_dtypes(include="number").columns.tolist()
    dep_var    = st.sidebar.selectbox("Dependent variable", numeric)
    indep_vars = st.sidebar.multiselect("Independent variables", numeric, default=numeric[:2])

    # FE selectors with “None”
    unit_fe = st.sidebar.selectbox("Unit fixed effect", ["None", "OD Pair"])
    time_fe = st.sidebar.selectbox("Time fixed effect", ["None", "Year"])

    if st.sidebar.button("Run Regression"):
        if not dep_var or not indep_vars:
            st.error("Please pick a dependent and ≥1 independent variable.")
        else:
            try:
                # 1) Clean column names
                cols_to_clean = [dep_var] + indep_vars + ([unit_fe] if unit_fe!="None" else []) + ([time_fe] if time_fe!="None" else [])
                rename_map = {}
                for col in cols_to_clean:
                    if col!="None":
                        clean = re.sub(r"\W+", "_", col)
                        rename_map[col] = clean
                reg_df = df.rename(columns=rename_map)

                # 2) Build formula
                dv = rename_map[dep_var]
                ivs = [rename_map[v] for v in indep_vars]
                fe_terms = []
                if unit_fe!="None":
                    fe_terms.append(f"C({rename_map[unit_fe]})")
                if time_fe!="None":
                    fe_terms.append(f"C({rename_map[time_fe]})")
                formula = dv + " ~ " + " + ".join(ivs + fe_terms)

                # 3) Fit OLS
                ols_res = smf.ols(formula, data=reg_df).fit()

                # 4) Attempt cluster‐robust if unit FE selected
                final_res = ols_res
                if unit_fe!="None":
                    groups = reg_df[rename_map[unit_fe]]
                    try:
                        final_res = ols_res.get_robustcov_results(
                            cov_type="cluster",
                            groups=groups
                        )
                    except Exception:
                        st.warning("Cluster‐robust failed; showing plain OLS instead.")

                # 5) Show summary
                st.subheader("Regression Results")
                st.text(final_res.summary().as_text())

            except Exception as e:
                st.error(f"Regression failed: {e}")

else:
    # ─────────────────────────────────────────────────────────────────
    # CROSS-SECTIONAL SIMULATOR MODE
    # ─────────────────────────────────────────────────────────────────
    st.header("📈 Cross-Section Simulator")

    # optional coords uploader
    coord_file = st.sidebar.file_uploader("2) Upload coords (.xlsx)", type=["xlsx"])

    # validate
    required = {
      "Origin Country Name","Destination Country Name",
      "Origin Airport","Destination Airport",
      "Distance (km)","Passengers","Avg. Total Fare(USD)"
    }
    if not required.issubset(df.columns):
        st.error("CSV missing required OD columns.")
        st.stop()
    df = df.dropna(subset=required).reset_index(drop=True)
    origin_all = sorted(df["Origin Country Name"].unique())
    dest_all   = sorted(df["Destination Country Name"].unique())

    # carbon pricing
    st.sidebar.markdown("### Carbon Pricing")
    enable_carbon = st.sidebar.checkbox("Enable carbon pricing?")
    if enable_carbon:
        ets_price = st.sidebar.slider("EUR / tCO₂", 0, 400, 100, 5)
        c_orig = st.sidebar.multiselect("Carbon origin", origin_all, default=origin_all)
        c_dest = st.sidebar.multiselect("Carbon dest",   dest_all,   default=dest_all)
    else:
        ets_price, c_orig, c_dest = 0.0, [], []

    # passenger tax
    st.sidebar.markdown("### Passenger Tax")
    enable_tax = st.sidebar.checkbox("Enable passenger tax?")
    if enable_tax:
        tax_amt = st.sidebar.slider("USD per pax", 0, 100, 10, 1)
        t_orig  = st.sidebar.multiselect("Tax origin", origin_all, default=origin_all)
        t_dest  = st.sidebar.multiselect("Tax dest",   dest_all,   default=dest_all)
    else:
        tax_amt, t_orig, t_dest = 0.0, [], []

    # parameters
    st.sidebar.markdown("### Parameters")
    pass_through    = st.sidebar.slider("Cost pass-through (%)", 0,100,80,5)/100
    emission_factor = st.sidebar.slider("Emission kg CO₂/pax-km", 0.0,1.0,0.115,0.001)

    # econ inputs
    st.sidebar.markdown("### Econ Inputs")
    global_gdp   = st.sidebar.slider("Global GDP growth (%)", -5.0,8.0,2.5,0.1)
    price_elast  = st.sidebar.slider("Price elasticity", -2.0,-0.1,PRICE_ELASTICITY_DEMAND,0.1)
    gdp_elast    = st.sidebar.slider("GDP elasticity", 0.5,2.0,GDP_ELASTICITY_DEMAND,0.1)

    # optional per-country GDP
    st.sidebar.markdown("### Optional: Country GDP")
    gdp_by_cty = {}
    with st.sidebar.expander("Adjust GDP growth by origin"):
        for c in origin_all:
            gdp_by_cty[c] = st.slider(f"{c} GDP (%)", -5.0,8.0,global_gdp,0.1)

    # -- simulate policy impacts --
    df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor
    df["Carbon cost per pax"] = 0.0
    if enable_carbon:
        m = df["Origin Country Name"].isin(c_orig)&df["Destination Country Name"].isin(c_dest)
        df.loc[m,"Carbon cost per pax"] = df.loc[m,"CO2 per pax (kg)"]/1000*ets_price*pass_through

    df["Air passenger tax per pax"] = 0.0
    if enable_tax:
        m = df["Origin Country Name"].isin(t_orig)&df["Destination Country Name"].isin(t_dest)
        df.loc[m,"Air passenger tax per pax"] = tax_amt*pass_through

    df["New Avg Fare"]    = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"] + df["Air passenger tax per pax"]
    df["Fare Δ (%)"]      = (df["New Avg Fare"]/df["Avg. Total Fare(USD)"] - 1)*100

    ff = (df["New Avg Fare"]/df["Avg. Total Fare(USD)"]).replace([np.inf,-np.inf], np.nan)**price_elast
    df["GDP Growth (%)"]    = df["Origin Country Name"].map(gdp_by_cty).fillna(global_gdp)
    df["GDP Growth Factor"] = (1+df["GDP Growth (%)"]/100)**gdp_elast
    df["Passengers after policy"] = df["Passengers"]*ff*df["GDP Growth Factor"]
    df["Passenger Δ (%)"]         = (df["Passengers after policy"]/df["Passengers"] - 1)*100

    # init coords
    for col in ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]:
        df[col] = np.nan

    # optional coords merge
    if coord_file:
        try:
            cd = pd.read_excel(coord_file, engine="openpyxl").drop_duplicates("IATA_Code")
            if {"IATA_Code","DecLat","DecLon"}.issubset(cd.columns):
                cmap = cd.set_index("IATA_Code")[["DecLat","DecLon"]]
                df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
                df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
                df["Origin Lat"]  = df["Origin Code"].map(cmap["DecLat"])
                df["Origin Lon"]  = df["Origin Code"].map(cmap["DecLon"])
                df["Dest Lat"]    = df["Dest Code"].map(cmap["DecLat"])
                df["Dest Lon"]    = df["Dest Code"].map(cmap["DecLon"])
            else:
                st.sidebar.warning("Coords need IATA_Code, DecLat, DecLon")
        except ImportError:
            st.sidebar.warning("Install openpyxl (`pip install openpyxl`)")
        except Exception as e:
            st.sidebar.warning(f"Coords load failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    # OUTPUTS: tables, bar charts, density, Kepler map (unchanged)
    # ─────────────────────────────────────────────────────────────────
    st.subheader("📊 Airport-Pair Results")
    st.dataframe(df[[
        "Origin Airport","Destination Airport","Passengers",
        "Distance (km)","CO2 per pax (kg)",
        "Avg. Total Fare(USD)","Carbon cost per pax",
        "Air passenger tax per pax","New Avg Fare","Passenger Δ (%)"
    ]], use_container_width=True)

    # … (the rest of your cross-sectional bar charts, density plot, Kepler map as before) …

