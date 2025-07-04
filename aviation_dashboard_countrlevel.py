import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Model configuration (defaults)
# ----------------------
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND = 1.4

# ----------------------
# Helper – generate dummy data when no CSV is provided
# ----------------------
def load_dummy_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    origins = ["Germany", "France", "United States", "Japan"]
    destinations = ["Spain", "Italy", "United Kingdom", "Canada"]
    rows = []
    for o in origins:
        for d in destinations:
            if o == d:
                continue
            rows.append({
                "Origin Country Name": o,
                "Destination Country Name": d,
                "Origin Airport": f"{o[:3].upper()}-INTL",
                "Destination Airport": f"{d[:3].upper()}-INTL",
                "Distance (km)": int(rng.integers(500, 9000)),
                "Passengers": int(rng.integers(50000, 1000000)),
                "Avg. Total Fare(USD)": round(rng.uniform(150, 700), 2),
            })
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between airports and policy impacts.")

st.sidebar.header("📈 Policy & Economic Inputs")
uploaded_file = st.sidebar.file_uploader("Upload airport-pair passenger CSV", type=["csv"])

# ----------------------
# Load data – CSV if provided, otherwise dummy
# ----------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded.")
else:
    df = load_dummy_data()
    st.info("🛈 No file uploaded – using **dummy data** to explore the simulator.")

# Validate columns
expected_columns = {
    "Origin Country Name", "Destination Country Name",
    "Origin Airport", "Destination Airport",
    "Distance (km)", "Passengers", "Avg. Total Fare(USD)"
}
if not expected_columns.issubset(df.columns):
    st.error(
        "CSV must contain columns: Origin Country Name, Destination Country Name, "
        "Origin Airport, Destination Airport, Distance (km), Passengers, Avg. Total Fare(USD)"
    )
    st.stop()
df = df.dropna(subset=expected_columns)
df["Avg. Total Fare(USD)"] = df["Avg. Total Fare(USD)"].fillna(0.0)

origin_all = sorted(df["Origin Country Name"].unique())
dest_all = sorted(df["Destination Country Name"].unique())

# ----------------------
# Carbon Pricing Policy
# ----------------------
st.sidebar.markdown("### Carbon Pricing Policy")
carbon_policy = st.sidebar.selectbox(
    "Enable carbon pricing?",
    ["Disable", "Enable"]
)
if carbon_policy == "Enable":
    ets_price = st.sidebar.slider(
        "Carbon price (EUR / tCO₂)",
        0, 400, 100, 5,
        help="Select the carbon price applied to aviation emissions."
    )
    carbon_origin_countries = st.sidebar.multiselect(
        "Origin countries taxed:",
        origin_all,
        default=origin_all
    )
    carbon_dest_countries = st.sidebar.multiselect(
        "Destination countries taxed:",
        dest_all,
        default=dest_all
    )
else:
    ets_price = 0.0
    carbon_origin_countries = []
    carbon_dest_countries = []

# ----------------------
# Parameters
# ----------------------
st.sidebar.markdown("### Parameters")

# Air Passenger Tax Policy
tax_policy = st.sidebar.selectbox(
    "Enable air passenger tax?",
    ["Disable", "Enable"]
)
if tax_policy == "Enable":
    air_passenger_tax = st.sidebar.slider(
        "Air Passenger Tax (USD)",
        0, 100, 0, 1,
        help="Flat additional tax added to each ticket fare."
    )
    tax_origin_countries = st.sidebar.multiselect(
        "Origin countries taxed:",
        origin_all,
        default=origin_all
    )
    tax_dest_countries = st.sidebar.multiselect(
        "Destination countries taxed:",
        dest_all,
        default=dest_all
    )
else:
    air_passenger_tax = 0.0
    tax_origin_countries = []
    tax_dest_countries = []

# Cost pass-through to fares (applies to both CO₂ price & tax)
pass_through = st.sidebar.slider(
    "Cost pass-through to fares (%)",
    0, 100, 80, 5,
    help="Share of carbon cost and ticket tax airlines embed in ticket prices."
) / 100

# Emission factor slider
emission_factor = st.sidebar.slider(
    "Emission factor (kg CO₂ per pax-km)",
    0.0, 1.0, 0.115, 0.001,
    help="Kilograms of CO₂ emitted per passenger-kilometer flown."
)

# ----------------------
# Other Economic Inputs
# ----------------------
global_gdp_growth = st.sidebar.slider(
    "Global Real GDP growth year-on-year (%)",
    -5.0, 8.0, 2.5, 0.1,
    help="Default GDP growth applied to all unless customized."
)
user_price_elasticity = st.sidebar.slider(
    "Demand price-elasticity (negative)",
    -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1,
    help="Passenger response to fare increases."
)
user_gdp_elasticity = st.sidebar.slider(
    "Demand GDP-elasticity",
    0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1,
    help="Passenger response to GDP growth."
)

st.sidebar.markdown("### Optional: Adjust GDP Growth by Origin Country")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth for Specific Countries"):
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

# Carbon cost per pax
df["Carbon cost per pax"] = 0.0
if carbon_policy == "Enable":
    mask_c = (
        df["Origin Country Name"].isin(carbon_origin_countries) &
        df["Destination Country Name"].isin(carbon_dest_countries)
    )
    df.loc[mask_c, "Carbon cost per pax"] = (
        df.loc[mask_c, "CO2 per pax (kg)"] / 1000
        * ets_price * pass_through
    )

# Air passenger tax per pax (passed through)
df["Air passenger tax per pax"] = 0.0
if tax_policy == "Enable":
    mask_t = (
        df["Origin Country Name"].isin(tax_origin_countries) &
        df["Destination Country Name"].isin(tax_dest_countries)
    )
    df.loc[mask_t, "Air passenger tax per pax"] = air_passenger_tax * pass_through

# New fare including policies
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

# ----------------------
# Output
# ----------------------
st.subheader("📊 Passenger Simulation Results (Airport-Pair Level)")
st.dataframe(
    df[[
        "Origin Airport", "Destination Airport",
        "Origin Country Name", "Destination Country Name",
        "Passengers", "Distance (km)", "CO2 per pax (kg)",
        "Avg. Total Fare(USD)", "Carbon cost per pax",
        "Air passenger tax per pax", "New Avg Fare",
        "Passengers after policy", "Passenger Δ (%)"
    ]],
    use_container_width=True
)

# Aggregate by origin country
origin_summary = df.groupby("Origin Country Name", as_index=False).agg({
    "Passengers": "sum",
    "Passengers after policy": "sum",
    "CO2 per pax (kg)": "mean",
})
origin_summary["Relative Change (%)"] = (
    origin_summary["Passengers after policy"] /
    origin_summary["Passengers"] - 1
) * 100

fig = px.bar(
    origin_summary,
    x="Origin Country Name",
    y="Relative Change (%)",
    title="Relative Change in Passenger Volume by Origin Country (%)",
    labels={"Relative Change (%)": "Change in Passengers (%)"},
)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    base_total = df["Passengers"].sum()
    new_total = df["Passengers after policy"].sum()
    delta_pct = (new_total / base_total - 1) * 100
    st.metric(
        "Total Passengers (m)",
        f"{new_total/1e6:,.2f} M",
        delta=f"{delta_pct:+.1f}% vs base",
    )
with col2:
    avg_cc = df["Carbon cost per pax"].mean()
    st.metric("Avg Carbon Cost per Ticket", f"€{avg_cc:.2f}")

st.info("💡 Each country inherits the global GDP growth unless adjusted manually.")
st.caption("Data: Sabre MI (or dummy) · Visualization powered by Streamlit & Plotly")
