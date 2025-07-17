import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from keplergl import KeplerGl
import streamlit.components.v1 as components
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf   # for regression demo
from streamlit_keplergl import keplergl_static

# ----------------------
# Model configuration (defaults)
# ----------------------
PRICE_ELASTICITY_DEMAND = -1.0
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
            if o == d:
                continue
            for _ in range(rng.integers(4,7)):
                rows.append({
                    "Origin Country Name":      o,
                    "Destination Country Name": d,
                    "Origin Airport":           f"{o[:3].upper()}-INTL",
                    "Destination Airport":      f"{d[:3].upper()}-INTL",
                    "Distance (km)":            int(rng.integers(500, 9000)),
                    "Passengers":               int(rng.integers(50_000, 1_000_000)),
                    "Avg. Total Fare(USD)":     round(rng.uniform(150, 700), 2),
                    "Total Revenue(USD)":       round(rng.uniform(1500, 7000), 2),
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
                "Origin Airport":               r["Origin Airport"],
                "Destination Airport":          r["Destination Airport"],
                "Operating Airline":            a,
                "Operating Airline   Capacity": float(rng.integers(5_000, 200_000)),
            })
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI setup
# ----------------------
st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Makes your life easier")

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

# ----------------------
# Detect which aggregation levels & fields are present
# ----------------------
has_airports   = {"Origin Airport", "Destination Airport"}.issubset(df.columns)
has_operating  = has_airports and ("Operating Airline" in df.columns)
has_distance   = "Distance (km)" in df.columns

# ----------------------
# Validate required columns for country‐level
# ----------------------
country_required = {
    "Origin Country Name", "Destination Country Name",
    "Passengers", "Avg. Total Fare(USD)"
}
if not country_required.issubset(df.columns):
    missing = country_required - set(df.columns)
    st.error(f"Passenger CSV missing required columns: {missing}")
    st.stop()

# For airport‐level, require airports; for distance‐based metrics, require distance
if has_airports:
    df = df.dropna(subset=["Origin Airport", "Destination Airport"])
    if has_distance:
        df = df.dropna(subset=["Distance (km)"])
    else:
        st.sidebar.warning("No `Distance (km)`—distance‐based metrics disabled.")
else:
    st.sidebar.info("No airport columns—airport‐level analyses disabled.")

df = df.reset_index(drop=True)

# Convert to str so sorting never fails on weird types
origin_all = sorted(df["Origin Country Name"].dropna().astype(str).unique())
dest_all   = sorted(df["Destination Country Name"].dropna().astype(str).unique())
panel_data = "Year" in df.columns

# ----------------------
# Default policy parameters
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
# Coordinates setup (only if airports exist)
# ----------------------
df["Origin Lat"] = np.nan; df["Origin Lon"] = np.nan
df["Dest Lat"]   = np.nan; df["Dest Lon"]   = np.nan

if has_airports:
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

    if {"IATA_Code", "DecLat", "DecLon"}.issubset(coords.columns):
        cmap = coords.set_index("IATA_Code")[["DecLat", "DecLon"]]
        df["Origin Code"] = df["Origin Airport"].str.partition("-")[0]
        df["Dest Code"]   = df["Destination Airport"].str.partition("-")[0]
        df["Origin Lat"]  = df["Origin Code"].map(cmap["DecLat"])
        df["Origin Lon"]  = df["Origin Code"].map(cmap["DecLon"])
        df["Dest Lat"]    = df["Dest Code"].map(cmap["DecLat"])
        df["Dest Lon"]    = df["Dest Code"].map(cmap["DecLon"])
    else:
        st.sidebar.warning("Coords missing required columns – using dummy coords")
else:
    st.sidebar.info("Skipping coordinates – no airport data available")

# ----------------------
# Main area by mode
# ----------------------
if mode == "Descriptives":
    tab_desc_me, tab_desc_sup = st.tabs(["Market Equilibrium", "Supply"])
    with tab_desc_me:
        st.subheader("📈/📊 Passenger numbers and airfares")
        col1, col2 = st.columns(2)
        with col1:
            metric    = st.selectbox("Metric", ["Passengers", "Avg. Total Fare(USD)","Total Revenue(USD)"], key="desc_metric")
        with col2:    
            plot_type = st.selectbox("Plot type", ["Line", "Bar"], key="desc_plot")
        is_long   = ("Year" in df.columns) or ("Month" in df.columns)
        has_operating = "Operating Airline" in df.columns
        has_airports = "Origin Airport" in df.columns

        # build group‐by levels dynamically
        group_levels = ["Origin Country Name","Destination Country Name"]
        if has_airports:
            group_levels.insert(0, "Origin Airport")
        if has_operating:
            group_levels.insert(0, "Operating Airline")

        if plot_type == "Line":
            if not is_long:
                st.warning("No Year/Month columns—cannot time‐series.")
            else:

                col1, col2 = st.columns(2)

                # put one selectbox in each
                with col1:
                    freq  = st.selectbox("Time frequency", ["Year", "Year-Month"], key="desc_freq")
                    agg   = st.selectbox("Aggregation", ["sum", "mean"], key="desc_agg")
                with col2:
                    level = st.selectbox("Group by", group_levels, key="desc_level")
                    top_n = st.number_input("Top N series", 1, 50, 10, key="desc_n")

                d = df.copy()
                if freq == "Year-Month" and "Month" in d.columns:
                    d["Year-Month"] = (
                        d["Year"].astype(str).str.zfill(4)
                        + "-"
                        + d["Month"].astype(str).str.zfill(2)
                    )
                    time_col = "Year-Month"
                else:
                    time_col = "Year"

                d = d.groupby([time_col, level], as_index=False)[metric].agg(agg)
                top_series = d.groupby(level)[metric].sum().nlargest(top_n).index
                d = d[d[level].isin(top_series)]

                fig = px.line(
                    d,
                    x=time_col,
                    y=metric,
                    color=level,
                    markers=True,
                    
                )
                
                fig.update_layout(
                    font=dict(size=20),               # base font size for title & legend
                    legend=dict(font=dict(size=20)),  # legend text size
                )
                
                fig.update_xaxes(
                    title_font_size=20,  # x‑axis title
                    tickfont_size=20     # x‑axis tick labels
                )
                fig.update_yaxes(
                    title_font_size=20,  # y‑axis title
                    tickfont_size=20     # y‑axis tick labels
                )

                
                st.plotly_chart(fig, use_container_width=True)

        else:  # Bar
            with col1:
                agg   = st.selectbox("Aggregation", ["sum", "mean"], key="desc_agg_cs")
                top_n = st.number_input("Top N", 1, 50, 10, key="desc_n_cs")
            with col2:
                level = st.selectbox("Group by", group_levels, key="desc_level_cs")
                

            d = (
                df.groupby(level, as_index=False)[metric]
                  .agg(agg)
                  .nlargest(top_n, metric)
            )
            fig = px.bar(
                d, x=level, y=metric, title=f"Cross-sectional {metric}"
            )
            st.plotly_chart(fig, use_container_width=True)


        

        st.subheader("🥧 Relative Passenger numbers and Total Revenue")

        col1, col2 = st.columns(2)

                # put one selectbox in each
            
        with col1:
            # choose metric and aggregation
            metric = st.selectbox(
                "Metric",
                ["Passengers", "Total Revenue(USD)"],
                key="rel_metric"
            )
            agg = st.selectbox(
                "Aggregation",
                ["sum", "mean"],
                key="rel_agg"
            )
        with col2:
            # select which origin countries to analyze
            all_origins = sorted(df["Origin Country Name"].dropna().unique())
            selected_origins = st.multiselect(
                "Select origin country(ies)",
                all_origins,
                default=all_origins[:5],
                key="rel_origins"
            )
    
            # build the possible aggregation‐levels,
            # always falling back to country‐level if no finer detail exists
            has_operating = "Operating Airline" in df.columns
            rel_levels = []
            if has_operating:
                rel_levels.append("Operating Airline")
            if has_airports:
                rel_levels.append("Origin Airport")
            # when only country‐to‐country data is present, group by destination country
            rel_levels.append("Destination Country Name")
        with col2:
            level = st.selectbox("Aggregation level", rel_levels, key="rel_level")
    
        if not selected_origins:
            st.warning("Please select at least one origin country.")
        else:
            cols = st.columns(len(selected_origins))
            for col, origin in zip(cols, selected_origins):
                with col:
                    df_o = df[df["Origin Country Name"] == origin]
    
                    # 1) aggregate at the chosen level
                    d_rel = df_o.groupby(level, as_index=False)[metric].agg(agg)
                    d_rel = d_rel.sort_values(metric, ascending=False)
    
                    # 2) pick top 9, lump the rest into "Others"
                    if len(d_rel) > 9:
                        top9 = d_rel.nlargest(9, metric)
                        others_sum = d_rel.loc[~d_rel[level].isin(top9[level]), metric].sum()
                        d_plot = pd.concat([
                            top9,
                            pd.DataFrame({level: ["Others"], metric: [others_sum]})
                        ], ignore_index=True)
                    else:
                        d_plot = d_rel.copy()
    
                    # 3) compute percentage shares
                    d_plot["Pct"] = d_plot[metric] / d_plot[metric].sum() * 100
    
                    # 4) render as a donut‐style pie chart
                    fig = px.pie(
                        d_plot,
                        names=level,
                        values="Pct",
                        title=f"{origin}",
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
        

        # Sankey: Passenger Flows 
        if "Origin Country Name" in df.columns:
            st.markdown("---")
            st.subheader("🔀 Sankey Passenger Flows")
            
            col1, col2 = st.columns(2)

                
            has_year = "Year" in df.columns
            if has_year:
                years = sorted(df["Year"].unique())

                with col1:
                    year = st.selectbox("Year", years, index=len(years) - 1)
                df_sel = df[df["Year"] == year]
            else:
                df_sel = df
        
            sankey_opts = ["Country"]
            if has_airports:
                sankey_opts.insert(0, "Airport")
                with col1:
                    agg_level = st.selectbox("Aggregation", sankey_opts)
        
            origin_col = (
                "Origin Airport" if agg_level == "Airport"
                else "Origin Country Name"
            )
            dest_col = (
                "Destination Airport" if agg_level == "Airport"
                else "Destination Country Name"
            )
        
            df_year = (
                df_sel
                .dropna(subset=[origin_col, dest_col])
                .groupby([origin_col, dest_col], as_index=False)["Passengers"]
                .sum()
            )
        
            all_origins = sorted(df_year[origin_col].unique())
            with col2:
                selected_origins = st.multiselect(
                    f"Select {agg_level.lower()}s of origin",
                    all_origins,
                    default=all_origins[:5],
                )
            with col2:    
                top_n_dest = st.number_input(
                    "Top N destinations per origin", 1, 50, 5, 1
            )
        
            parts = []
            for orig in selected_origins:
                sub = df_year[df_year[origin_col] == orig]
                if not sub.empty:
                    parts.append(sub.nlargest(top_n_dest, "Passengers"))
            flows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        
            if flows.empty:
                st.warning("No flows to display—adjust selections.")
            else:
                labels = list(dict.fromkeys(
                    flows[origin_col].tolist() + flows[dest_col].tolist()
                ))
                idx = {lab: i for i, lab in enumerate(labels)}
                src = flows[origin_col].map(idx).tolist()
                tgt = flows[dest_col].map(idx).tolist()
                vals = flows["Passengers"].tolist()
        
                palette = px.colors.qualitative.Plotly
                unique_o = list(dict.fromkeys(flows[origin_col].tolist()))
                color_map = {o: palette[i % len(palette)] for i, o in enumerate(unique_o)}
                link_colors = []
                for o in flows[origin_col]:
                    hexc = color_map[o].lstrip("#")
                    r, g, b = (int(hexc[i:i+2], 16) for i in (0, 2, 4))
                    link_colors.append(f"rgba({r},{g},{b},0.4)")
        
                sankey = go.Sankey(
                    arrangement="freeform",
                    node=dict(label=labels, pad=15, thickness=20),
                    link=dict(source=src, target=tgt, value=vals, color=link_colors)
                )
                fig = go.Figure(data=[sankey])

                fig.update_traces(
                    textfont_shadow="none",
                    textfont_color="black",
                    selector=dict(type="sankey")
                    )
                
                fig.update_layout(
                    font=dict(size=18),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add an `Origin Country Name` column to enable the Sankey diagram.")
 

    
    


    
    with tab_desc_sup:
        st.subheader("📦 Descriptive: Supply Data (Dummy)")
        if has_airports:
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
                title="Total Capacity by Origin Airport",
            )
            st.plotly_chart(fig_sup, use_container_width=True)
        else:
            st.info("Supply analysis requires airport‐level data.")

elif mode == "Simulation":
    
    st.subheader("⚙️ Whats the scenario?")
    
    if has_distance:
        enable_carbon = st.checkbox("Carbon pricing")
    
        if enable_carbon:
            ets_price = st.slider("Carbon price (EUR/tCO₂)", 0, 400, 100, 5)
            carbon_origin = st.multiselect("Carbon taxed: Origin countries", origin_all, default=origin_all)
            carbon_dest   = st.multiselect("Carbon taxed: Destination countries", dest_all,   default=dest_all)
    else:
        st.sidebar.info("No distance, no carbon simulation")

    enable_tax = st.checkbox("Air passenger tax")
    if enable_tax:
        air_pass_tax = st.slider("Air Passenger Tax (USD)", 0, 100, 10, 1)
        tax_origin   = st.multiselect("Taxed: Origin countries", origin_all, default=origin_all)
        tax_dest     = st.multiselect("Taxed: Destination countries", dest_all,   default=dest_all)

    enable_freak = st.checkbox("Trump freaks out")
    if enable_freak: 
         st.header("🍊 🍊 🍊")
        
    

        # ----------------------
    # Apply policies to data
    # ----------------------
    # CO2 & carbon cost only if distance exists
    if has_distance:
        df["CO2 per pax (kg)"]    = df["Distance (km)"] * emission_factor
        df["Carbon cost per pax"] = 0.0
        if mode != "Descriptives" and enable_carbon:
            mask_c = (
                df["Origin Country Name"].isin(carbon_origin)
                & df["Destination Country Name"].isin(carbon_dest)
            )
            df.loc[mask_c, "Carbon cost per pax"] = (
                df.loc[mask_c, "CO2 per pax (kg)"] / 1000 * ets_price * pass_through
            )
    else:
        df["CO2 per pax (kg)"]    = 0.0
        df["Carbon cost per pax"] = 0.0
    
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

    if enable_freak:
         df["Fare Δ (%)"] = 100000
        
    fare_factor = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"]) ** price_elast
    df["GDP Growth (%)"]    = df["Origin Country Name"].map(gdp_by_country).fillna(global_gdp)
    df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"]/100) ** gdp_elast
    
    df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
    df["Passenger Δ (%)"]         = (df["Passengers after policy"] / df["Passengers"] - 1) * 100
    
        
    sub1, sub2 = st.tabs(["Direct effects", "Catalytic effects"])
    with sub1:
        tab_sim_me, tab_sim_sup = st.tabs(["Market Equilibrium", "Supply"])
        with tab_sim_me:
            # choose ID columns
            if has_airports:
                id_orig, id_dest = "Origin Airport", "Destination Airport"
            else:
                id_orig, id_dest = "Origin Country Name", "Destination Country Name"

            agg_options = ["Origin Country Name"]
            if has_airports:   agg_options.insert(0, "Origin Airport")
            if has_operating:  agg_options.insert(0, "Operating Airline")
            
            # build table columns
            table_cols = []
            if has_operating:
                table_cols.append("Operating Airline")
            table_cols += [id_orig, id_dest]

            # include distance/CO2 only if available
            metrics = ["Passengers"]
            if has_distance:
                metrics += ["Distance (km)", "CO2 per pax (kg)"]
            metrics += [
                "Avg. Total Fare(USD)",
                "Carbon cost per pax",
                "Air passenger tax per pax",
                "New Avg Fare",
                "Passenger Δ (%)",
            ]
            st.dataframe(df[table_cols + metrics], use_container_width=True)

            level = st.selectbox("Aggregation Level", agg_options)
            
            # Bar chart: passenger change by origin
            ps = df.groupby(level, as_index=False).agg(
                Passengers=("Passengers","sum"),
                After=("Passengers after policy","sum")
            )
            ps["Δ (%)"] = (ps["After"]/ps["Passengers"] - 1)*100
            fig1 = px.bar(
                ps, x=level, y="Δ (%)", text="Δ (%)",
                title=f"Passenger Change by {level}"
            )
            fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

            fig1.update_layout(
                    font=dict(size=20),               # base font size for title & legend
                    legend=dict(font=dict(size=20)),  # legend text size
                )
                
                
            fig1.update_xaxes(
                    title_font_size=20,  # x‑axis title
                    tickfont_size=20     # x‑axis tick labels
                )
                
            fig1.update_yaxes(
                    title_font_size=20,  # y‑axis title
                    tickfont_size=20     # y‑axis tick labels
                )

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
            # Average fare change: average Fare Δ (%) at level
            pf = df.groupby(level, as_index=False).agg(
                **{"Avg Δ (%)":("Fare Δ (%)","mean")}
            )
            fig2 = px.bar(
                pf, x=level, y="Avg Δ (%)", text="Avg Δ (%)",
                title=f"Average Fare Change by {level}"
            )
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)
            

            # Distance KDE scaled by passenger count (only if distance exists)
            if has_distance:
                df_b = df[["Distance (km)", "Passengers"]].rename(columns={"Distance (km)": "x", "Passengers": "w"})
                df_a = df[["Distance (km)", "Passengers after policy"]].rename(columns={"Distance (km)": "x", "Passengers after policy": "w"})
                scenarios = {"Before": df_b, "After": df_a}
                fig3 = go.Figure()
                for name, sub in scenarios.items():
                    xvals = sub["x"].dropna().to_numpy()
                    wvals = sub["w"].fillna(0).to_numpy()
                    if len(xvals) < 2 or wvals.sum() <= 0:
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
            else:
                st.info("Distance‐based density plot disabled (no `Distance (km)` column).")

            # Kepler map: country‐level arcs (requires airports + coords)
            if has_airports and not df[["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]].isna().all().all():
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
                       .rename(columns={"Lat":"A Lat","Lon":"A Lon"}).drop(columns="Country")
                pa = pa.merge(cents, left_on="B", right_on="Country") \
                       .rename(columns={"Lat":"B Lat","Lon":"B Lon"}).drop(columns="Country")
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
                            "colorScale":"diverging",
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
                      "zoom":3,"pitch":30
                    },
                    "mapStyle": {}
                  }
                }
                map1 = KeplerGl(height=800, width=1600, data={"pairs": pa}, config=cfg)
                keplergl_static(
                    map1,
                    height=800,
                    width=1600,
                    center_map=False,
                    read_only=False
                )
            else:
                if has_airports:
                    st.warning("Upload coords to see Kepler map.")
                else:
                    st.info("Airport‐level coordinates not available – map disabled.")

        with tab_sim_sup:
            st.subheader("📦 Supply-side HHI & Capacity Share Analysis")
            if has_airports:
                supply_file = st.file_uploader("Upload supply CSV", type=["csv"], key="supply")
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
            else:
                st.info("Supply analysis requires airport‐level data.")

    with sub2:
        st.subheader("🧪 Catalytic effects")
        origin_dim = "Origin Airport" if has_airports else "Origin Country Name"
        airport_df = df.groupby(origin_dim, as_index=False).agg(
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
            hover_name=origin_dim,
            title=f"Catalytic Effects: Passenger Change vs Regional GDP Change by {origin_dim}",
            labels={
                "Change": "Passenger Change",
                "GDP Change": "Regional GDP Change",
                "Passengers": "Total Passengers"
            }
        )
        fig_bubble.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig_bubble, use_container_width=True)

elif mode == "Regression":
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
