import re
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
st.set_page_config(page_title="✈️ JETPAS Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Use the tabs below to switch between the **Simulator** and **Regression** modes.")

# ----------------------
# Upload OD data
# ----------------------
uploaded_file = st.sidebar.file_uploader(
    "1) Upload airport-pair passenger CSV", type=["csv"]
)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ CSV loaded.")
else:
    df = load_dummy_data()
    st.sidebar.info("🛈 No CSV – using dummy data.")

# Detect panel vs cross-sectional
is_panel = "Year" in df.columns

# Create tabs
sim_tab, reg_tab = st.tabs(["Simulator", "Regression"])

with sim_tab:
    st.header("📈 Cross-Section Simulator")
    # ----------------------
    # Optional coords uploader
    # ----------------------
    coord_file = st.sidebar.file_uploader(
        "2) Upload airport coordinates (.xlsx)", type=["xlsx"]
    )

    # ----------------------
    # Validate passenger columns
    # ----------------------
    required = {
        "Origin Country Name", "Destination Country Name",
        "Origin Airport",        "Destination Airport",
        "Distance (km)",         "Passengers",
        "Avg. Total Fare(USD)"
    }
    if not required.issubset(df.columns):
        st.error("Passenger CSV missing required columns.")
        st.stop()
    df_cs = df.dropna(subset=required).reset_index(drop=True)

    origin_all = sorted(df_cs["Origin Country Name"].unique())
    dest_all   = sorted(df_cs["Destination Country Name"].unique())

    # ----------------------
    # Carbon pricing policy
    # ----------------------
    st.sidebar.markdown("### Carbon Pricing Policy")
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
    st.sidebar.markdown("### Air Passenger Tax Policy")
    enable_tax = st.sidebar.checkbox("Enable air passenger tax?")
    if enable_tax:
        air_passenger_tax = st.sidebar.slider(
            "Air Passenger Tax (USD)", 0, 100, 10, 1
        )
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
    st.sidebar.markdown("### Economic Inputs")
    global_gdp_growth = st.sidebar.slider("Global real GDP growth (%)", -5.0, 8.0, 2.5, 0.1)
    user_price_elasticity = st.sidebar.slider(
        "Demand price-elasticity (negative)", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1
    )
    user_gdp_elasticity = st.sidebar.slider(
        "Demand GDP-elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1
    )

    st.sidebar.markdown("### Optional: Adjust GDP Growth by Origin Country")
    gdp_growth_by_country = {}
    with st.sidebar.expander("Customize GDP Growth for Specific Origins"):
        for country in origin_all:
            gdp_growth_by_country[country] = st.slider(
                f"{country} GDP growth (%)",
                -5.0, 8.0, global_gdp_growth, 0.1,
                key=f"gdp_{country}"
            )

    # ----------------------
    # Deep copy for simulation
    # ----------------------
    df = df_cs.copy()

    # ----------------------
    # Policy calculations
    # ----------------------
    df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor

    df["Carbon cost per pax"] = 0.0
    if enable_carbon:
        mask_c = (
            df["Origin Country Name"].isin(carbon_origin_countries)
            & df["Destination Country Name"].isin(carbon_dest_countries)
        )
        df.loc[mask_c, "Carbon cost per pax"] = (
            df.loc[mask_c, "CO2 per pax (kg)"] / 1000
            * ets_price * pass_through
        )

    df["Air passenger tax per pax"] = 0.0
    if enable_tax:
        mask_t = (
            df["Origin Country Name"].isin(tax_origin_countries)
            & df["Destination Country Name"].isin(tax_dest_countries)
        )
        df.loc[mask_t, "Air passenger tax per pax"] = air_passenger_tax * pass_through

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
    # Initialize coords for Kepler
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
                df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
                df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
                df["Origin Lat"]  = df["Origin Code"].map(coords_map["DecLat"])
                df["Origin Lon"]  = df["Origin Code"].map(coords_map["DecLon"])
                df["Dest Lat"]    = df["Dest Code"].map(coords_map["DecLat"])
                df["Dest Lon"]    = df["Dest Code"].map(coords_map["DecLon"])
            else:
                st.warning("❌ Coordinate file missing IATA_Code / DecLat / DecLon.")
        except ImportError:
            st.warning("❌ Install 'openpyxl' to read .xlsx files: pip install openpyxl")
        except Exception as e:
            st.warning(f"❌ Failed to process coordinate file: {e}")

    # ----------------------
    # Outputs
    # ----------------------
    st.subheader("📊 Airport-Pair Passenger Results")
    st.dataframe(
        df[[
            "Origin Airport", "Destination Airport", "Passengers",
            "Distance (km)", "CO2 per pax (kg)",
            "Avg. Total Fare(USD)", "Carbon cost per pax",
            "Air passenger tax per pax", "New Avg Fare",
            "Passengers after policy", "Passenger Δ (%)"
        ]],
        use_container_width=True
    )

    # Barplot: passenger change by origin country
    origin_summary = df.groupby("Origin Country Name", as_index=False).agg({
        "Passengers":              "sum",
        "Passengers after policy": "sum"
    })
    origin_summary["Relative Change (%)"] = (
        origin_summary["Passengers after policy"] /
        origin_summary["Passengers"] - 1
    ) * 100

    fig1 = px.bar(
        origin_summary,
        x="Origin Country Name",
        y="Relative Change (%)",
        title="📉 Relative Change in Passenger Volume by Origin Country",
        text="Relative Change (%)",
        labels={"Relative Change (%)": "Δ Passengers (%)"}
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        base = df["Passengers"].sum()
        new  = df["Passengers after policy"].sum()
        st.metric(
            "Total Passengers (M)",
            f"{new/1e6:,.2f}",
            delta=f"{(new/base-1)*100:+.1f}%"
        )
    with col2:
        st.metric(
            "Avg. Carbon Cost per Ticket (€)",
            f"{df['Carbon cost per pax'].mean():.2f}"
        )

    # Barplot: fare change by origin country
    origin_price_summary = df.groupby("Origin Country Name", as_index=False).agg({
        "Fare Δ (%)": "mean"
    }).rename(columns={"Fare Δ (%)": "Avg Fare Δ (%)"})

    fig2 = px.bar(
        origin_price_summary,
        x="Origin Country Name",
        y="Avg Fare Δ (%)",
        title="📈 Relative Change in Average Fare by Origin Country",
        text="Avg Fare Δ (%)",
        labels={"Avg Fare Δ (%)": "Δ Fare (%)"}
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # Density plot: distance distribution
    df_before = df[["Distance (km)", "Passengers"]].rename(
        columns={"Distance (km)": "Distance_km", "Passengers": "Count"}
    )
    df_after = df[["Distance (km)", "Passengers after policy"]].rename(
        columns={"Distance (km)": "Distance_km", "Passengers after policy": "Count"}
    )
    # build smoothed densities
    min_d, max_d = df_before["Distance_km"].min(), df_before["Distance_km"].max()
    min_d = min(min_d, df_after["Distance_km"].min())
    max_d = max(max_d, df_after["Distance_km"].max())
    bins = np.linspace(min_d, max_d, 50)
    dens_list = []
    for label, subset in [("Before", df_before), ("After", df_after)]:
        x = subset["Distance_km"].to_numpy()
        w = subset["Count"].to_numpy()
        hist, edges = np.histogram(x, bins=bins, weights=w, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dens_list.append(pd.DataFrame({
            "Distance (km)": centers,
            "Density": hist,
            "Scenario": label
        }))
    dens_df = pd.concat(dens_list, ignore_index=True)
    fig3 = px.line(
        dens_df,
        x="Distance (km)",
        y="Density",
        color="Scenario",
        title="📊 Passenger Distance Density: Before vs After",
        labels={"Density": "Density", "Distance (km)": "Distance (km)"}
    )
    fig3.update_traces(line_shape="spline")
    st.plotly_chart(fig3, use_container_width=True)

    # Kepler map: country‐level arcs
    required_centroid_cols = ["Origin Lat", "Origin Lon", "Dest Lat", "Dest Lon"]
    if all(col in df.columns for col in required_centroid_cols):
        # compute centroids
        coords_orig = df[["Origin Country Name","Origin Lat","Origin Lon"]].rename(
            columns={"Origin Country Name":"Country","Origin Lat":"Lat","Origin Lon":"Lon"}
        )
        coords_dest = df[["Destination Country Name","Dest Lat","Dest Lon"]].rename(
            columns={"Destination Country Name":"Country","Dest Lat":"Lat","Dest Lon":"Lon"}
        )
        centroids = pd.concat([coords_orig, coords_dest],ignore_index=True) \
                      .dropna(subset=["Lat","Lon"]) \
                      .groupby("Country",as_index=False)[["Lat","Lon"]].mean()

        # aggregate unordered country-pairs
        ab = df[["Origin Country Name","Destination Country Name","Passengers","Passengers after policy"]].copy()
        ab["A"] = np.where(
            ab["Origin Country Name"] < ab["Destination Country Name"],
            ab["Origin Country Name"],
            ab["Destination Country Name"]
        )
        ab["B"] = np.where(
            ab["Origin Country Name"] < ab["Destination Country Name"],
            ab["Destination Country Name"],
            ab["Origin Country Name"]
        )
        pair_agg = ab.groupby(["A","B"],as_index=False).agg({
            "Passengers":"sum","Passengers after policy":"sum"
        })
        pair_agg["Traffic Δ (%)"] = (
            pair_agg["Passengers after policy"] / pair_agg["Passengers"] - 1
        ) * 100

        # merge centroids
        pair_agg = pair_agg.merge(
            centroids, left_on="A", right_on="Country", how="left"
        ).rename(columns={"Lat":"A Lat","Lon":"A Lon"}).drop(columns=["Country"])
        pair_agg = pair_agg.merge(
            centroids, left_on="B", right_on="Country", how="left"
        ).rename(columns={"Lat":"B Lat","Lon":"B Lon"}).drop(columns=["Country"])

        # Kepler config
        kepler_config = {
            "version":"v1","config":{
                "visState":{
                    "filters": [],
                    "layers":[{
                        "id":"arc_layer","type":"arc","config":{
                            "dataId":"pairs","label":"Traffic Δ (%)",
                            "columns":{"lat0":"A Lat","lng0":"A Lon","lat1":"B Lat","lng1":"B Lon"},
                            "isVisible":True,
                            "visConfig":{
                                "thickness":3,"opacity":0.8,
                                "colorField":{"name":"Traffic Δ (%)","type":"real"},
                                "colorScale":"quantile",
                                "colorRange":{"name":"Global Warming","type":"sequential",
                                              "category":"Uber",
                                              "colors":["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]},
                                "sizeField":"Traffic Δ (%)","sizeScale":10
                            }
                        }
                    }],
                    "interactionConfig":{
                        "tooltip":{"fieldsToShow":{"pairs":["A","B","Traffic Δ (%)"]},
                                   "enabled":True}
                    }
                },
                "mapState":{
                    "latitude":centroids["Lat"].mean(),
                    "longitude":centroids["Lon"].mean(),
                    "zoom":2.2,"pitch":30
                },
                "mapStyle":{}
            }
        }

        kepler_map = KeplerGl(height=1600, data={"pairs":pair_agg}, config=kepler_config)
        raw_html = kepler_map._repr_html_()
        if isinstance(raw_html, bytes):
            raw_html = raw_html.decode("utf-8")
        components.html(raw_html, height=1200, width=1800)
    else:
        st.warning(
            "Upload coordinates with 'IATA_Code', 'DecLat', 'DecLon' → "
            "Origin/Dest Lat/Lon to see the Kepler map."
        )

with reg_tab:
    st.header("📊 Panel Regression Analysis")
    if not is_panel:
        st.warning("No `Year` column detected in your data – regression disabled.")
    else:
        st.success("Detected panel data (Year present).")
        # ensure OD Pair identifier
        if "OD Pair" not in df.columns:
            df["OD Pair"] = df["Origin Airport"] + " ↔ " + df["Destination Airport"]

        # Sidebar regression controls
        st.sidebar.header("Regression Settings")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        dep_var    = st.sidebar.selectbox("Dependent variable", numeric_cols)
        indep_vars = st.sidebar.multiselect("Independent variables", numeric_cols, default=numeric_cols[:2])
        unit_fe    = st.sidebar.selectbox("Unit fixed effect", ["None", "OD Pair"])
        time_fe    = st.sidebar.selectbox("Time fixed effect", ["None", "Year"])
        if st.sidebar.button("Run Regression"):
            if not dep_var or not indep_vars:
                st.error("Pick a dependent and at least one independent variable.")
            else:
                try:
                    # clean column names
                    to_clean = [dep_var] + indep_vars + ([unit_fe] if unit_fe!="None" else []) + ([time_fe] if time_fe!="None" else [])
                    rename_map = {}
                    for col in to_clean:
                        if col != "None":
                            clean = re.sub(r"\W+", "_", col)
                            rename_map[col] = clean
                    reg_df = df.rename(columns=rename_map)

                    # build formula
                    dv = rename_map[dep_var]
                    ivs = [rename_map[v] for v in indep_vars]
                    fe_terms = []
                    if unit_fe!="None":
                        fe_terms.append(f"C({rename_map[unit_fe]})")
                    if time_fe!="None":
                        fe_terms.append(f"C({rename_map[time_fe]})")
                    formula = dv + " ~ " + " + ".join(ivs + fe_terms)

                    # fit OLS
                    ols_res = smf.ols(formula, data=reg_df).fit()

                    # cluster‐robust if OD Pair FE
                    final_res = ols_res
                    if unit_fe=="OD Pair":
                        try:
                            final_res = ols_res.get_robustcov_results(
                                cov_type="cluster",
                                groups=reg_df[rename_map["OD Pair"]]
                            )
                        except Exception:
                            st.warning("Cluster‐robust failed; showing plain OLS.")

                    st.subheader("Regression Results")
                    st.text(final_res.summary().as_text())
                except Exception as e:
                    st.error(f"Regression failed: {e}")
