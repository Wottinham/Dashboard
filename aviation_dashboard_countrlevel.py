import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
from keplergl import KeplerGl
import streamlit.components.v1 as components
import statsmodels.formula.api as smf

# ─── Model defaults ─────────────────────────────────────────────────────────────
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND   = 1.4

# ─── Dummy data helper ───────────────────────────────────────────────────────────
def load_dummy_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    origins = ["Germany","France","United States","Japan"]
    dests   = ["Spain","Italy","United Kingdom","Canada"]
    rows = []
    for o in origins:
        for d in dests:
            if o==d: continue
            rows.append({
                "Origin Country Name":      o,
                "Destination Country Name": d,
                "Origin Airport":           f"{o[:3].upper()}-INTL",
                "Destination Airport":      f"{d[:3].upper()}-INTL",
                "Distance (km)":            int(rng.integers(500,9000)),
                "Passengers":               int(rng.integers(50_000,1_000_000)),
                "Avg. Total Fare(USD)":     float(round(rng.uniform(150,700),2)),
            })
    return pd.DataFrame(rows)

# ─── UI setup ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="JETPAS Simulator", layout="wide")
st.title("✈️ JETPAS – Joint Economic & Transport Policy Simulator")

# Uploads
st.sidebar.header("📈 Policy & Data Inputs")
uploaded_file = st.sidebar.file_uploader("Passenger CSV", type="csv")
coord_file    = st.sidebar.file_uploader("Airport Coords (.xlsx)", type="xlsx")

# Load passenger data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Passenger CSV loaded")
else:
    df = load_dummy_data()
    st.info("🛈 Using dummy data")

# Detect cross‐section vs panel
is_panel = "Year" in df.columns

# Validate
req = {
    "Origin Country Name","Destination Country Name",
    "Origin Airport","Destination Airport",
    "Distance (km)","Passengers","Avg. Total Fare(USD)"
}
if not req.issubset(df.columns):
    st.error("Missing required passenger columns")
    st.stop()
df = df.dropna(subset=req).reset_index(drop=True)

# Sidebar: carbon & tax
enable_c = st.sidebar.checkbox("Enable carbon pricing?")
if enable_c:
    ets_price = st.sidebar.slider("Carbon price (€ / tCO₂)", 0,400,100,5)
    carbon_orig = st.sidebar.multiselect("Taxed origins (C)", sorted(df["Origin Country Name"].unique()),
                                         default=sorted(df["Origin Country Name"].unique()))
    carbon_dest = st.sidebar.multiselect("Taxed dests (C)",  sorted(df["Destination Country Name"].unique()),
                                         default=sorted(df["Destination Country Name"].unique()))
else:
    ets_price = 0.0; carbon_orig=carbon_dest=[]

enable_t = st.sidebar.checkbox("Enable air passenger tax?")
if enable_t:
    tax_val = st.sidebar.slider("Passenger tax (USD)", 0,100,10,1)
    tax_orig = st.sidebar.multiselect("Taxed origins (T)", sorted(df["Origin Country Name"].unique()),
                                       default=sorted(df["Origin Country Name"].unique()))
    tax_dest = st.sidebar.multiselect("Taxed dests (T)",  sorted(df["Destination Country Name"].unique()),
                                       default=sorted(df["Destination Country Name"].unique()))
else:
    tax_val=0.0; tax_orig=tax_dest=[]

# Other parameters
st.sidebar.markdown("### Parameters")
pass_through    = st.sidebar.slider("Cost pass‐through (%)",0,100,80,5)/100
emission_factor = st.sidebar.slider("Emission factor (kg CO₂ per pax-km)",0.0,1.0,0.115,0.001)
gdp_global      = st.sidebar.slider("Global GDP growth (%)",-5.0,8.0,2.5,0.1)
price_elast     = st.sidebar.slider("Price elasticity",-2.0,-0.1,PRICE_ELASTICITY_DEMAND,0.1)
gdp_elast       = st.sidebar.slider("GDP elasticity",0.5,2.0,GDP_ELASTICITY_DEMAND,0.1)

# Optional GDP per origin
gdp_by_country = {}
with st.sidebar.expander("Adjust GDP growth by origin"):
    for c in sorted(df["Origin Country Name"].unique()):
        gdp_by_country[c] = st.slider(f"{c} GDP (%)",-5.0,8.0,gdp_global,0.1)

# ─── Compute Simulation columns (always) ────────────────────────────────────────
df["CO2 per pax (kg)"] = df["Distance (km)"] * emission_factor

# carbon cost
df["Carbon cost per pax"] = 0.0
if enable_c:
    m = df["Origin Country Name"].isin(carbon_orig)&df["Destination Country Name"].isin(carbon_dest)
    df.loc[m,"Carbon cost per pax"] = df.loc[m,"CO2 per pax (kg)"]/1000 * ets_price * pass_through

# tax cost
df["Air passenger tax per pax"] = 0.0
if enable_t:
    m = df["Origin Country Name"].isin(tax_orig)&df["Destination Country Name"].isin(tax_dest)
    df.loc[m,"Air passenger tax per pax"] = tax_val * pass_through

# new fare & Δ
df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"] + df["Air passenger tax per pax"]
df["Fare Δ (%)"] = (df["New Avg Fare"]/df["Avg. Total Fare(USD)] - 1)*100

# elasticity & GDP effects
fare_factor = (df["New Avg Fare"]/df["Avg. Total Fare(USD)"]).replace([np.inf,-np.inf],np.nan)**price_elast
df["GDP Growth (%)"] = df["Origin Country Name"].map(gdp_by_country).fillna(gdp_global)
df["GDP Growth Factor"] = (1+df["GDP Growth (%)"]/100)**gdp_elast

df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"] = (df["Passengers after policy"]/df["Passengers"] - 1)*100

# coords placeholders
for c in ("Origin Lat","Origin Lon","Dest Lat","Dest Lon"): df[c]=np.nan

# load coords
if coord_file:
    try:
        cd = pd.read_excel(coord_file,engine="openpyxl")
        cd = cd.drop_duplicates("IATA_Code")
        if {"IATA_Code","DecLat","DecLon"}.issubset(cd.columns):
            cmap = cd.set_index("IATA_Code")[["DecLat","DecLon"]]
            df["OrigCode"]=df["Origin Airport"].str.split("-",1).str[0]
            df["DestCode"]=df["Destination Airport"].str.split("-",1).str[0]
            df["Origin Lat"]=df["OrigCode"].map(cmap["DecLat"])
            df["Origin Lon"]=df["OrigCode"].map(cmap["DecLon"])
            df["Dest Lat"]=df["DestCode"].map(cmap["DecLat"])
            df["Dest Lon"]=df["DestCode"].map(cmap["DecLon"])
        else:
            st.warning("❌ coord file missing IATA_Code/DecLat/DecLon")
    except ImportError:
        st.warning("❌ install openpyxl to read .xlsx")
    except Exception as e:
        st.warning(f"❌ Failed to process coords: {e}")

# ─── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Simulation","Regression"])

# ─── Simulation Tab ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Airport‐Pair Passenger Results")
    st.dataframe(df[[
        "Origin Airport","Destination Airport","Passengers",
        "Distance (km)","CO2 per pax (kg)","Avg. Total Fare(USD)",
        "Carbon cost per pax","Air passenger tax per pax",
        "New Avg Fare","Passenger Δ (%)"
    ]], use_container_width=True)

    # 1) Δ passengers by origin
    origin_summary = df.groupby("Origin Country Name",as_index=False).agg({
        "Passengers":"sum","Passengers after policy":"sum"
    })
    origin_summary["Relative Change (%)"] = (
        origin_summary["Passengers after policy"]/origin_summary["Passengers"]-1
    )*100

    fig1 = px.bar(origin_summary,
                  x="Origin Country Name", y="Relative Change (%)",
                  text="Relative Change (%)",
                  title="Δ Passenger Volume by Origin")
    fig1.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
    st.plotly_chart(fig1, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        base, new = df["Passengers"].sum(), df["Passengers after policy"].sum()
        st.metric("Total Passengers (M)",f"{new/1e6:,.2f}",delta=f"{(new/base-1)*100:+.1f}%")
    with c2:
        st.metric("Avg Carbon Cost (€)",f"{df['Carbon cost per pax'].mean():.2f}")

    st.markdown("---")

    # 2) Δ fare by origin
    price_summary = df.groupby("Origin Country Name",as_index=False)["Fare Δ (%)"].mean().rename({"Fare Δ (%)":"Avg Fare Δ (%)"},axis=1)
    fig2 = px.bar(price_summary, x="Origin Country Name",y="Avg Fare Δ (%)",
                  text="Avg Fare Δ (%)", title="Δ Avg Fare by Origin")
    fig2.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # 3) Density curves, scaled by total pax
    df_b = df[["Distance (km)","Passengers"]].rename(columns={"Distance (km)":"Distance_km","Passengers":"Count"})
    df_a = df[["Distance (km)","Passengers after policy"]].rename(columns={"Distance (km)":"Distance_km","Passengers after policy":"Count"})
    x_min = min(df_b["Distance_km"].min(),df_a["Distance_km"].min())
    x_max = max(df_b["Distance_km"].max(),df_a["Distance_km"].max())
    x_grid = np.linspace(x_min,x_max,500)

    dens_dfs=[]
    for label, sub in [("Before",df_b),("After",df_a)]:
        x=sub["Distance_km"].values; w=sub["Count"].values
        try:
            kde = gaussian_kde(x,weights=w)
            y = kde(x_grid)*w.sum()
            dens_dfs.append(pd.DataFrame({"Distance (km)":x_grid,"Count":y,"Scenario":label}))
        except Exception:
            # fallback to simple histogram
            hist,edges=np.histogram(x,bins=50,weights=w)
            centers=0.5*(edges[:-1]+edges[1:])
            dens_dfs.append(pd.DataFrame({"Distance (km)":centers,"Count":hist,"Scenario":label}))

    dens_df=pd.concat(dens_dfs,ignore_index=True)
    fig3=px.line(dens_df,x="Distance (km)",y="Count",color="Scenario",
                 title="Passenger Distance Distribution: Before vs After")
    fig3.update_traces(line_shape="spline")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # 4) Kepler country-level arcs
    reqc=["Origin Lat","Origin Lon","Dest Lat","Dest Lon"]
    if all(col in df.columns for col in reqc):
        # centroids
        ocd=df[["Origin Country Name","Origin Lat","Origin Lon"]].rename(columns={
            "Origin Country Name":"Country","Origin Lat":"Lat","Origin Lon":"Lon"})
        dcd=df[["Destination Country Name","Dest Lat","Dest Lon"]].rename(columns={
            "Destination Country Name":"Country","Dest Lat":"Lat","Dest Lon":"Lon"})
        cent=pd.concat([ocd,dcd],ignore_index=True).dropna(subset=["Lat","Lon"])
        cent=cent.groupby("Country",as_index=False)[["Lat","Lon"]].mean()

        # unordered pairs
        ab=df[["Origin Country Name","Destination Country Name","Passengers","Passengers after policy"]].copy()
        ab["A"]=np.where(ab["Origin Country Name"]<ab["Destination Country Name"],
                         ab["Origin Country Name"],ab["Destination Country Name"])
        ab["B"]=np.where(ab["Origin Country Name"]<ab["Destination Country Name"],
                         ab["Destination Country Name"],ab["Origin Country Name"])
        pa=ab.groupby(["A","B"],as_index=False).agg({"Passengers":"sum","Passengers after policy":"sum"})
        pa["Traffic Δ (%)"]=(pa["Passengers after policy"]/pa["Passengers"]-1)*100

        # merge centroids
        pa=pa.merge(cent,left_on="A", right_on="Country", how="left").rename({"Lat":"A Lat","Lon":"A Lon"},axis=1).drop("Country",axis=1)
        pa=pa.merge(cent,left_on="B", right_on="Country", how="left").rename({"Lat":"B Lat","Lon":"B Lon"},axis=1).drop("Country",axis=1)

        cfg={
          "version":"v1","config":{
            "visState":{
              "layers":[{
                "id":"arcs","type":"arc","config":{
                  "dataId":"pairs","label":"Traffic Δ (%)",
                  "columns":{"lat0":"A Lat","lng0":"A Lon","lat1":"B Lat","lng1":"B Lon"},
                  "isVisible":True,
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
            "mapState":{"latitude":cent["Lat"].mean(),"longitude":cent["Lon"].mean(),
                        "zoom":2.2,"pitch":30},
            "mapStyle":{}
          }
        }

        km=KeplerGl(height=1600,data={"pairs":pa},config=cfg)
        raw=km._repr_html_()
        if isinstance(raw,bytes): raw=raw.decode()
        components.html(raw,height=1200,width=1800,scrolling=True)
    else:
        st.info("Upload coords to see Kepler map")

# ─── Regression Tab ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Panel Regression Analysis")
    if not is_panel:
        st.info("Upload a CSV with a `Year` column to enable panel regression.")
    else:
        # define OD pair
        df["OD_Pair"]=df["Origin Airport"]+" ─ "+df["Destination Airport"]

        # select vars
        num_cols = df.select_dtypes(include="number").columns.tolist()
        dep = st.selectbox("Dependent var", num_cols, index= num_cols.index("Passengers after policy") if "Passengers after policy" in num_cols else 0)
        indep = st.multiselect("Independent vars", [c for c in num_cols if c!=dep], default=["Fare Δ (%)","Passenger Δ (%)"][:2])

        fe_time = st.checkbox("Include Year fixed effects?", value=True)
        fe_unit = st.checkbox("Include OD_Pair fixed effects?", value=False)
        run = st.button("Run regression")

        if run:
            # build formula
            rhs = indep.copy()
            if fe_time: rhs.append("C(Year)")
            if fe_unit: rhs.append("C(OD_Pair)")
            formula = f"{dep} ~ " + " + ".join(rhs)
            try:
                model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups":df["OD_Pair"]})
                st.write(model.summary())
            except Exception as e:
                st.error(f"Regression failed: {e}")
