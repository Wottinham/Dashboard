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
tab_sim, tab_desc, tab_reg = st.tabs(["Descriptives","Simulation", "Regression"])
# ---- Descriptives tab ----
with tab_desc:
    desc_me, desc_sup = st.tabs(["Market Equilibrium", "Supply"])

    # -- Descriptives → Market Equilibrium --
    with desc_me:
        st.subheader("📈 Descriptive: Passenger Flow")

        # detect if longitudinal (Year or Month present)
        is_long = ("Year" in df.columns) or ("Month" in df.columns)

        if is_long:
            view = st.radio("View type", ["Longitudinal", "Cross-sectional"], key="desc_view")
            if view == "Longitudinal":
                freq = st.selectbox("Time frequency", ["Year", "Year-Month"], key="desc_freq")
            agg   = st.selectbox("Aggregation", ["sum", "mean"], key="desc_agg")
            level = st.selectbox(
                "Group by", 
                ["Origin Airport", "Origin Country Name"], 
                key="desc_level"
            )
            top_n = st.number_input("Top N series", min_value=1, max_value=50, value=10, key="desc_n")

            d = df.copy()

            # build time column
            if view == "Longitudinal":
                if (freq == "Year-Month") and ("Month" in d.columns):
                    d["Year-Month"] = d["Year"].astype(str) + "-" + d["Month"].astype(str).str.zfill(2)
                    time_col = "Year-Month"
                else:
                    time_col = "Year"
                group_cols = [time_col, level]
            else:
                time_col = None
                group_cols = [level]

            # aggregate passengers
            agg_func = "sum" if agg == "sum" else "mean"
            d = d.groupby(group_cols, as_index=False)["Passengers"].agg(agg_func)

            # keep only top-N series
            if view == "Longitudinal" and time_col:
                d = (
                    d
                    .groupby(time_col, group_keys=False)
                    .apply(lambda g: g.nlargest(top_n, "Passengers"))
                    .reset_index(drop=True)
                )
            else:
                d = d.nlargest(top_n, "Passengers")

            # plot
            if time_col:
                fig = px.line(
                    d,
                    x=time_col,
                    y="Passengers",
                    color=level,
                    markers=True,
                    title="Passenger Flow over Time"
                )
            else:
                fig = px.bar(
                    d,
                    x=level,
                    y="Passengers",
                    title="Cross-sectional Passenger Volumes"
                )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # pure cross-sectional if no Year/Month
            agg   = st.selectbox("Aggregation", ["sum", "mean"], key="desc_agg_cs")
            level = st.selectbox("Group by", ["Origin Airport", "Origin Country Name"], key="desc_level_cs")
            top_n = st.number_input("Top N", min_value=1, max_value=50, value=10, key="desc_n_cs")

            d = (
                df
                .groupby(level, as_index=False)["Passengers"]
                .agg("sum" if agg=="sum" else "mean")
                .nlargest(top_n, "Passengers")
            )
            fig = px.bar(
                d,
                x=level,
                y="Passengers",
                title="Cross-sectional Passenger Volumes"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Descriptives → Supply --
    with desc_sup:
        st.subheader("📦 Descriptive: Supply Data")
        # (detection of cross-section vs longitudinal & similar controls go here)

# ---- Simulation tab ----
with tab_sim:
    sub1, sub2 = st.tabs(["Direct effects", "Catalytic effects"])

    # Direct effects with nested tabs
    with sub1:
        tab_me, tab_supply = st.tabs(["Market Equilibrium", "Supply"])

        # Market Equilibrium: current direct-effects analysis
        with tab_me:
            st.subheader("📊 Airport-Pair Passenger Results")
            st.dataframe(df[[
                "Origin Airport","Destination Airport","Passengers",
                "Distance (km)","CO2 per pax (kg)",
                "Avg. Total Fare(USD)","Carbon cost per pax",
                "Air passenger tax per pax","New Avg Fare",
                "Passenger Δ (%)"
            ]], use_container_width=True)

            # Bar chart: passenger change by origin
            origin_summary = df.groupby("Origin Country Name", as_index=False).agg(
                Passengers=("Passengers","sum"),
                After=("Passengers after policy","sum")
            )
            origin_summary["Δ (%)"] = (origin_summary["After"]/origin_summary["Passengers"]-1)*100
            fig1 = px.bar(
                origin_summary, x="Origin Country Name", y="Δ (%)",
                title="Passenger Change by Origin", text="Δ (%)",
                labels={"Δ (%)":"Δ Passengers (%)"}
            )
            fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig1, use_container_width=True)

            # Key metrics
            col1, col2 = st.columns(2)
            with col1:
                base_tot = df["Passengers"].sum()
                new_tot  = df["Passengers after policy"].sum()
                st.metric("Total Passengers", f"{new_tot:,.0f}",
                          delta=f"{(new_tot/base_tot-1)*100:+.1f}%")
            with col2:
                st.metric("Avg Carbon Cost (€)", f"{df['Carbon cost per pax'].mean():.2f}")

            # Bar chart: fare change
            price_summary = df.groupby("Origin Country Name", as_index=False).agg(
                **{"Avg Δ (%)":("Fare Δ (%)","mean")}
            )
            fig2 = px.bar(
                price_summary, x="Origin Country Name", y="Avg Δ (%)",
                title="Average Fare Change by Origin", text="Avg Δ (%)",
                labels={"Avg Δ (%)":"Δ Fare (%)"}
            )
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

            # Density KDE scaled by passenger count
            df_b = df[["Distance (km)","Passengers"]].rename(columns={"Distance (km)":"x","Passengers":"w"})
            df_a = df[["Distance (km)","Passengers after policy"]].rename(columns={"Distance (km)":"x","Passengers after policy":"w"})
            scenarios = {"Before":df_b, "After":df_a}
            fig3 = go.Figure()
            for name, sub in scenarios.items():
                xvals = sub["x"].dropna().to_numpy()
                wvals = sub["w"].fillna(0).to_numpy()
                if len(xvals)<2 or wvals.sum()<=0:
                    continue
                kde = gaussian_kde(xvals, weights=wvals)
                xs = np.linspace(xvals.min(), xvals.max(), 500)
                ys = kde(xs) * wvals.sum()
                fig3.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy", name=name))
            fig3.update_layout(
                title="Passenger Distance Density",
                xaxis_title="Distance (km)",
                yaxis_title="Passengers"
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Kepler map: country-level arcs (Δ %)
            cols_req = ["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]
            if all(c in df.columns for c in cols_req):
                orig = df[["Origin Country Name","Origin Lat","Origin Lon"]].rename(
                    columns={"Origin Country Name":"Country","Origin Lat":"Lat","Origin Lon":"Lon"})
                dest = df[["Destination Country Name","Dest Lat","Dest Lon"]].rename(
                    columns={"Destination Country Name":"Country","Dest Lat":"Lat","Dest Lon":"Lon"})
                cents = pd.concat([orig,dest],ignore_index=True).dropna(subset=["Lat","Lon"])\
                         .groupby("Country",as_index=False)[["Lat","Lon"]].mean()
                ab = df[["Origin Country Name","Destination Country Name","Passengers","Passengers after policy"]].copy()
                ab["A"] = np.where(ab["Origin Country Name"]<ab["Destination Country Name"],
                                   ab["Origin Country Name"], ab["Destination Country Name"])
                ab["B"] = np.where(ab["Origin Country Name"]<ab["Destination Country Name"],
                                   ab["Destination Country Name"], ab["Origin Country Name"])
                pa = ab.groupby(["A","B"],as_index=False).agg(
                    Passengers=("Passengers","sum"),
                    After=("Passengers after policy","sum")
                )
                pa["Δ (%)"] = (pa["After"]/pa["Passengers"]-1)*100
                pa = pa.merge(cents,left_on="A",right_on="Country").rename(
                    columns={"Lat":"A Lat","Lon":"A Lon"}).drop(columns="Country")
                pa = pa.merge(cents,left_on="B",right_on="Country").rename(
                    columns={"Lat":"B Lat","Lon":"B Lon"}).drop(columns="Country")
                cfg = {
                  "version":"v1","config":{
                    "visState":{
                      "filters":[],
                      "layers":[{
                        "id":"arc","type":"arc","config":{
                          "dataId":"pairs","label":"Δ (%)",
                          "columns":{
                            "lat0":"A Lat","lng0":"A Lon","lat1":"B Lat","lng1":"B Lon"
                          },
                          "isVisible":True,
                          "visConfig":{
                            "thickness":3,"opacity":0.8,
                            "colorField":{"name":"Δ (%)","type":"real"},
                            "colorScale":"quantile",
                            "colorRange":{
                              "name":"Global Warming","type":"sequential","category":"Uber",
                              "colors":["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]
                            },
                            "sizeField":"Δ (%)","sizeScale":10
                          }
                        }
                      }],
                      "interactionConfig":{
                        "tooltip":{
                          "fieldsToShow":{"pairs":["A","B","Δ (%)"]},
                          "enabled":True
                        }
                      }
                    },
                    "mapState":{
                      "latitude":cents["Lat"].mean(),
                      "longitude":cents["Lon"].mean(),
                      "zoom":2.2,"pitch":30
                    },
                    "mapStyle":{}
                  }
                }
                map1 = KeplerGl(height=600, data={"pairs":pa}, config=cfg)
                html = map1._repr_html_()
                if isinstance(html, bytes):
                    html = html.decode("utf-8")
                components.html(html, height=500)
            else:
                st.warning("Upload coords to see Kepler map.")

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
                    ).fillna({"Passenger Δ (%)": 0})

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

                                         # Pie charts: top-10 capacity share, wrapped into rows of up to 4 per row
                    countries = supply_df["Origin Country Name"].unique().tolist()
                    max_cols = 4
                    for i in range(0, len(countries), max_cols):
                        chunk = countries[i:i+max_cols]
                        cols = st.columns(len(chunk))
                        for col, country in zip(cols, chunk):
                            pie_df = (
                                supply_df[supply_df["Origin Country Name"] == country]
                                .groupby("Operating Airline", as_index=False)
                                .agg({"Adj Capacity":"sum"})
                                .nlargest(10, "Adj Capacity")
                            )
                            fig_pie = px.pie(
                                pie_df,
                                names="Operating Airline",
                                values="Adj Capacity",
                                title=f"{country}: Top 10 Airline Capacity Share",
                                height=300,
                                width=300
                            )
                            with col:
                                st.plotly_chart(fig_pie, use_container_width=False)


            else:
                st.info("Upload a supply CSV to see HHI & capacity share analysis.")

    # Catalytic effects
    with sub2:
        st.subheader("🧪 Catalytic effects")
        airport_df = df.groupby("Origin Airport", as_index=False).agg(
            Passengers=("Passengers","sum"),
            After=("Passengers after policy","sum")
        )
        airport_df["Change"]     = airport_df["After"] - airport_df["Passengers"]
        airport_df["GDP Change"] = 0.1 * airport_df["Change"]
        fig_bubble = px.scatter(
            airport_df,
            x="Change",
            y="GDP Change",
            size="Passengers",
            hover_name="Origin Airport",
            title="Catalytic Effects: Passenger Change vs Regional GDP Change",
            labels={
                "Change": "Passenger Change",
                "GDP Change": "Regional GDP Change",
                "Passengers": "Total Passengers"
            }
        )
        fig_bubble.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig_bubble, use_container_width=True)

# ---- Regression tab ----
with tab_reg:
    st.subheader("📊 Panel Regression Analysis")
    if panel_data:
        st.write("Detected 'Year' column – ready for regression.")
        # Regression UI and output go here...
    else:
        st.info("Upload panel data with a 'Year' column to enable regression mode.")
