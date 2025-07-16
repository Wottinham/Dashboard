import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from keplergl import KeplerGl
import streamlit.components.v1 as components
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf   # for regression demo

# ----------------------
# Model configuration (defaults)
# ----------------------
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND   = 1.4

# ----------------------
# Helper – generate dummy passenger data when no CSV is provided
# ----------------------
def load_dummy_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    origins = ["Germany", "France", "United States", "Japan"]
    dests   = ["Spain", "Italy", "United Kingdom", "Canada"]
    years   = list(range(2018, 2025))
    rows = []
    for o in origins:
        for d in dests:
            if o == d: continue
            # assign a random sample of 4–6 observations per pair
            for _ in range(rng.integers(4,7)):
                rows.append({
                    "Origin Country Name":      o,
                    "Destination Country Name": d,
                    "Origin Airport":           f"{o[:3].upper()}-INTL",
                    "Destination Airport":      f"{d[:3].upper()}-INTL",
                    "Distance (km)":            int(rng.integers(500, 9000)),
                    "Passengers":               int(rng.integers(50_000, 1_000_000)),
                    "Avg. Total Fare(USD)":     round(rng.uniform(150, 700), 2),
                    "Year":                     int(rng.choice(years)),
                    "Month":                    int(rng.integers(1, 13)),
                })
    return pd.DataFrame(rows)

# ----------------------
# Helper – generate dummy coords when no XLSX is provided
# ----------------------
def load_dummy_coords(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(seed=1)
    orig_codes = df["Origin Airport"].str.split("-", expand=True)[0]
    dest_codes = df["Destination Airport"].str.split("-", expand=True)[0]
    codes = pd.unique(pd.concat([orig_codes, dest_codes], ignore_index=True))
    rows = []
    for code in codes:
        rows.append({
            "IATA_Code": code,
            "DecLat":    float(rng.uniform(-60, 80)),
            "DecLon":    float(rng.uniform(-180, 180)),
        })
    return pd.DataFrame(rows)

# ----------------------
# Helper – generate dummy supply data when no CSV is provided
# ----------------------
def load_dummy_supply(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(seed=24)
    pairs = df[["Origin Airport","Destination Airport"]].drop_duplicates()
    airlines = ["AirFast","SkyHigh","GlobeAir","FlyRight","JetQuick"]
    rows = []
    for _, r in pairs.iterrows():
        n = rng.integers(2,5)
        chosen = rng.choice(airlines, size=n, replace=False)
        for a in chosen:
            rows.append({
                "Origin Airport":                r["Origin Airport"],
                "Destination Airport":           r["Destination Airport"],
                "Operating Airline":             a,
                "Operating Airline   Capacity":  float(rng.integers(5_000, 200_000)),
            })
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI setup
# ----------------------
st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between airports and policy impacts.")

# Sidebar – uploads
st.sidebar.header("💻 Data & Mode")
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
panel_data = "Year" in df.columns

# ----------------------
# Default policy parameters (so variables exist even in Descriptives mode)
# ----------------------
ets_price       = 0.0
carbon_origin   = []
carbon_dest     = []
air_pass_tax    = 0.0
tax_origin      = []
tax_dest        = []
pass_through    = 0.80
emission_factor = 0.115
global_gdp      = 2.5
price_elast     = PRICE_ELASTICITY_DEMAND
gdp_elast       = GDP_ELASTICITY_DEMAND
gdp_by_country  = {c: global_gdp for c in origin_all}

# ----------------------
# Top‐level view selector
# ----------------------
mode = st.sidebar.radio("Select mode", ["Descriptives", "Simulation", "Regression"])

# ----------------------
# Sidebar – policy & parameter controls (hidden in Descriptives)
# ----------------------
if mode != "Descriptives":
    enable_carbon = st.sidebar.checkbox("Enable carbon pricing")
    if enable_carbon:
        ets_price = st.sidebar.slider("Carbon price (EUR/tCO₂)", 0, 400, 100, 5)
        carbon_origin = st.sidebar.multiselect("Carbon taxed: Origin countries", origin_all, default=origin_all)
        carbon_dest   = st.sidebar.multiselect("Carbon taxed: Destination countries", dest_all,   default=dest_all)
    else:
        ets_price = 0.0
        carbon_origin = []
        carbon_dest   = []

    enable_tax = st.sidebar.checkbox("Enable air passenger tax")
    if enable_tax:
        air_pass_tax = st.sidebar.slider("Air Passenger Tax (USD)", 0, 100, 10, 1)
        tax_origin   = st.sidebar.multiselect("Taxed: Origin countries", origin_all, default=origin_all)
        tax_dest     = st.sidebar.multiselect("Taxed: Destination countries", dest_all,   default=dest_all)
    else:
        air_pass_tax = 0.0
        tax_origin   = []
        tax_dest     = []

    st.sidebar.markdown("### Parameters")
    pass_through    = st.sidebar.slider("Cost pass-through (%)", 0, 100, 80, 5) / 100
    emission_factor = st.sidebar.slider("Emission factor (kg CO₂/pax-km)", 0.0, 1.0, 0.115, 0.001)
    global_gdp      = st.sidebar.slider("Global GDP growth (%)", -5.0, 8.0, 2.5, 0.1)
    price_elast     = st.sidebar.slider("Price elasticity (neg)", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1)
    gdp_elast       = st.sidebar.slider("GDP elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1)

    st.sidebar.markdown("### Optional: GDP by origin")
    with st.sidebar.expander("Customize GDP growth"):
        for c in origin_all:
            gdp_by_country[c] = st.slider(
                f"{c} GDP growth (%)", -5.0, 8.0, global_gdp, 0.1, key=f"gdp_{c}"
            )

# ----------------------
# Apply policies to data
# ----------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor
df["Carbon cost per pax"] = 0.0
if mode != "Descriptives" and enable_carbon:
    mask_c = (
        df["Origin Country Name"].isin(carbon_origin)
        & df["Destination Country Name"].isin(carbon_dest)
    )
    df.loc[mask_c, "Carbon cost per pax"] = (
        df.loc[mask_c, "CO2 per pax (kg)"] / 1000 * ets_price * pass_through
    )

df["Air passenger tax per pax"] = 0.0
if mode != "Descriptives" and enable_tax:
    mask_t = (
        df["Origin Country Name"].isin(tax_origin)
        & df["Destination Country Name"].isin(tax_dest)
    )
    df.loc[mask_t, "Air passenger tax per pax"] = air_pass_tax * pass_through

df["New Avg Fare"] = (
    df["Avg. Total Fare(USD)"]
    + df["Carbon cost per pax"]
    + df["Air passenger tax per pax"]
)
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

fare_factor = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"]) ** price_elast
df["GDP Growth (%)"]    = df["Origin Country Name"].map(gdp_by_country).fillna(global_gdp)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"]/100) ** gdp_elast

df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# Initialize coords
df["Origin Lat"] = np.nan; df["Origin Lon"] = np.nan
df["Dest Lat"]   = np.nan; df["Dest Lon"]   = np.nan

# ----------------------
# Load & merge coordinates (dummy if none)
# ----------------------
if coord_file:
    try:
        coords = pd.read_excel(coord_file, engine="openpyxl").drop_duplicates("IATA_Code")
    except ImportError:
        st.sidebar.warning("Install openpyxl to read .xlsx")
        coords = load_dummy_coords(df)
    except Exception as e:
        st.sidebar.warning(f"Failed coords processing: {e}")
        coords = load_dummy_coords(df)
else:
    coords = load_dummy_coords(df)
    st.sidebar.info("No coords file – using dummy coords.")

if {"IATA_Code","DecLat","DecLon"}.issubset(coords.columns):
    cmap = coords.set_index("IATA_Code")[["DecLat","DecLon"]]
    df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
    df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
    df["Origin Lat"]  = df["Origin Code"].map(cmap["DecLat"])
    df["Origin Lon"]  = df["Origin Code"].map(cmap["DecLon"])
    df["Dest Lat"]    = df["Dest Code"].map(cmap["DecLat"])
    df["Dest Lon"]    = df["Dest Code"].map(cmap["DecLon"])
else:
    st.sidebar.warning("Coords missing required columns – using dummy coords")

# ----------------------
# Main area by mode
# ----------------------
if mode == "Descriptives":
    tab_desc_me, tab_desc_sup = st.tabs(["Market Equilibrium", "Supply"])
    with tab_desc_me:
        st.subheader("📈 Descriptive: Passenger Flow")
        metric    = st.selectbox("Metric", ["Passengers", "Avg. Total Fare(USD)"], key="desc_metric")
        plot_type = st.selectbox("Plot type", ["Line", "Bar"], key="desc_plot")
        is_long = ("Year" in df.columns) or ("Month" in df.columns)

        if plot_type == "Line":
            if not is_long:
                st.warning("Data has no Year/Month column – cannot do time series.")
            else:
                freq = st.selectbox("Time frequency", ["Year", "Year-Month"], key="desc_freq")
                agg   = st.selectbox("Aggregation", ["sum", "mean"], key="desc_agg")
                level = st.selectbox("Group by", ["Origin Airport", "Origin Country Name"], key="desc_level")
                top_n = st.number_input("Top N series", 1, 50, 10, key="desc_n")

                d = df.copy()
                if freq == "Year-Month" and "Month" in d.columns:
                    d["Year-Month"] = d["Year"].astype(str) + "-" + d["Month"].astype(str).str.zfill(2)
                    time_col = "Year-Month"
                else:
                    time_col = "Year"

                d = d.groupby([time_col, level], as_index=False)[metric].agg(agg)
                top_series = d.groupby(level)[metric].sum().nlargest(top_n).index
                d = d[d[level].isin(top_series)]

                fig = px.line(
                    d, x=time_col, y=metric, color=level,
                    markers=True, title=f"{metric} over Time"
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            agg   = st.selectbox("Aggregation", ["sum", "mean"], key="desc_agg_cs")
            level = st.selectbox("Group by", ["Origin Airport", "Origin Country Name"], key="desc_level_cs")
            top_n = st.number_input("Top N", 1, 50, 10, key="desc_n_cs")

            d = (
                df
                .groupby(level, as_index=False)[metric]
                .agg(agg)
                .nlargest(top_n, metric)
            )
            fig = px.bar(
                d, x=level, y=metric, title=f"Cross-sectional {metric}"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Sankey: Passenger Flows by Year
        if "Year" in df.columns:
            st.markdown("---")
            st.subheader("🔀 Sankey: Passenger Flows by Year")

            years      = sorted(df["Year"].unique())
            year       = st.selectbox("Year", years, index=len(years)-1)
            agg_level  = st.selectbox("Aggregation", ["Airport", "Country"])
            origin_col = "Origin Airport" if agg_level=="Airport" else "Origin Country Name"
            dest_col   = "Destination Airport" if agg_level=="Airport" else "Destination Country Name"

            all_origins      = sorted(df[origin_col].dropna().unique())
            selected_origins = st.multiselect(
                f"Select {agg_level.lower()}s of origin",
                all_origins,
                default=all_origins[:5]
            )
            top_n_dest = st.number_input(
                "Top N destinations per origin", 1, 50, 5, 1
            )

            df_year = (
                df[df["Year"] == year]
                  .dropna(subset=[origin_col, dest_col])
                  .groupby([origin_col, dest_col], as_index=False)["Passengers"]
                  .sum()
            )
            df_year = df_year[df_year[origin_col].isin(selected_origins)]

            parts = []
            for orig in selected_origins:
                sub = df_year[df_year[origin_col] == orig]
                if not sub.empty:
                    parts.append(sub.nlargest(top_n_dest, "Passengers"))
            flows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

            if flows.empty:
                st.warning("No flows to display—adjust your selections or try another year.")
            else:
                labels = list(dict.fromkeys(
                    flows[origin_col].tolist() + flows[dest_col].tolist()
                ))
                idx    = {lab: i for i, lab in enumerate(labels)}
                src    = flows[origin_col].map(idx).tolist()
                tgt    = flows[dest_col].map(idx).tolist()
                vals   = flows["Passengers"].tolist()

                palette      = px.colors.qualitative.Plotly
                unique_orig  = list(dict.fromkeys(flows[origin_col].tolist()))
                color_map    = {
                    orig: palette[i % len(palette)]
                    for i, orig in enumerate(unique_orig)
                }
                link_colors = []
                for orig in flows[origin_col]:
                    hexc = color_map[orig].lstrip("#")
                    r, g, b = (int(hexc[i:i+2], 16) for i in (0, 2, 4))
                    link_colors.append(f"rgba({r},{g},{b},0.4)")

                sankey = go.Sankey(
                    arrangement="snap",
                    node=dict(
                        label=labels,
                        pad=15,
                        thickness=20
                    ),
                    link=dict(
                        source=src,
                        target=tgt,
                        value=vals,
                        color=link_colors
                    )
                )
                fig = go.Figure(data=[sankey])
                fig.update_layout(
                    title_text=f"Passenger Flows in {year} ({agg_level}-level)",
                    font=dict(size=18)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add a `Year` column to your data to enable the Sankey diagram.")

    with tab_desc_sup:
        st.subheader("📦 Descriptive: Supply Data (Dummy)")
        supply_demo = load_dummy_supply(df)
        st.dataframe(supply_demo.head(), use_container_width=True)
        sup_sum = (
            supply_demo
            .groupby("Origin Airport", as_index=False)["Operating Airline   Capacity"]
            .sum()
        )
        fig_sup = px.bar(
            sup_sum,
            x="Origin Airport",
            y="Operating Airline   Capacity",
            title="Total Capacity by Origin Airport"
        )
        st.plotly_chart(fig_sup, use_container_width=True)

elif mode == "Simulation":
    sub1, sub2 = st.tabs(["Direct effects", "Catalytic effects"])
    with sub1:
        tab_sim_me, tab_sim_sup = st.tabs(["Market Equilibrium", "Supply"])
        with tab_sim_me:
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
                pa = ab.groupby(["A","B"], as_index=False).agg(
                    Passengers=("Passengers","sum"),
                    After=("Passengers after policy","sum")
                )
                pa["Δ (%)"] = (pa["After"]/pa["Passengers"] - 1) * 100
                pa = pa.merge(cents, left_on="A", right_on="Country") \
                       .rename(columns={"Lat":"A Lat","Lon":"A Lon"}) \
                       .drop(columns="Country")
                pa = pa.merge(cents, left_on="B", right_on="Country") \
                       .rename(columns={"Lat":"B Lat","Lon":"B Lon"}) \
                       .drop(columns="Country")
                cfg = {
                  "version":"v1","config":{
                    "visState":{"filters":[],"layers":[{
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
                    }],"interactionConfig":{
                      "tooltip":{
                        "fieldsToShow":{"pairs":["A","B","Δ (%)"]},
                        "enabled":True
                      }
                    }},
                    "mapState":{
                      "latitude":cents["Lat"].mean(),
                      "longitude":cents["Lon"].mean(),
                      "zoom":2.2,"pitch":30
                    },
                    "mapStyle":{}
                  }
                }
                map1 = KeplerGl(height=800, data={"pairs": pa}, config=cfg)
                html = map1._repr_html_()
                if isinstance(html, bytes):
                    html = html.decode("utf-8")

                components.html(html, height=800, width=1600)
            else:
                st.warning("Upload coords to see Kepler map.")


        with tab_sim_sup:
            st.subheader("📦 Supply-side HHI & Capacity Share Analysis")
            supply_file = st.file_uploader(
                "Upload supply CSV",
                type=["csv"],
                key="supply"
            )
            if supply_file:
                supply_df = pd.read_csv(supply_file)
            else:
                st.info("No supply CSV – using dummy supply data.")
                supply_df = load_dummy_supply(df)

            req_sup = {
                "Origin Airport","Destination Airport",
                "Operating Airline","Operating Airline   Capacity"
            }
            if not req_sup.issubset(supply_df.columns):
                st.error("Supply data missing required columns.")
            else:
                orig_map = df[["Origin Airport","Origin Country Name"]].drop_duplicates()
                dest_map = df[["Destination Airport","Destination Country Name"]].drop_duplicates()
                supply_df = supply_df.merge(orig_map, on="Origin Airport", how="left")
                supply_df = supply_df.merge(dest_map, on="Destination Airport", how="left")
                pass_change = df[[
                    "Origin Airport","Destination Airport","Passenger Δ (%)"
                ]]
                supply_df = supply_df.merge(
                    pass_change,
                    on=["Origin Airport","Destination Airport"],
                    how="left"
                ).fillna({"Passenger Δ (%)": 0})
                supply_df["Adj Capacity"] = (
                    supply_df["Operating Airline   Capacity"]
                    * (1 + supply_df["Passenger Δ (%)"] / 100)
                )

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
                fig_hhi = px.box(
                    hhi,
                    x="Origin Country Name",
                    y="HHI",
                    title="HHI per Airport Pair by Origin Country (Adjusted Capacity)",
                    labels={"HHI":"HHI Index"}
                )
                st.plotly_chart(fig_hhi, use_container_width=True)

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

else:  # Regression
    st.subheader("📊 Panel Regression Analysis")
    if panel_data:
        st.write("Detected 'Year' column – running demo regression…")
        model = smf.ols(
            formula="Passengers ~ Q('New Avg Fare') + Q('CO2 per pax (kg)')",
            data=df
        ).fit()
        st.text(model.summary())
    else:
        st.info("Upload panel data with a 'Year' column to enable regression mode.")
