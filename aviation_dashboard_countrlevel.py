import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Model configuration
# ----------------------

EMISSION_FACTOR = 0.115  # kg CO₂ per pax-km
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
            rows.append(
                {
                    "Origin Country Name": o,
                    "Destination Country Name": d,
                    "Origin Airport": f"{o[:3].upper()}-INTL",
                    "Destination Airport": f"{d[:3].upper()}-INTL",
                    "Distance (km)": int(rng.integers(500, 9000)),
                    "Passengers": int(rng.integers(50000, 1000000)),
                    "Avg. Total Fare(USD)": round(rng.uniform(150, 700), 2),
                }
            )
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="Airport-Pair Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between airports.")

uploaded_file = st.sidebar.file_uploader(
    "Upload airport-pair passenger CSV", type=["csv"]
)

# ----------------------
# Load data – CSV if provided, otherwise dummy
# ----------------------

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded.")
else:
    df = load_dummy_data()
    st.info("🛈 No file uploaded – using **dummy data** to explore the simulator.")

# ----------------------
# Validate / clean data
# ----------------------

expected_columns = {
    "Origin Country Name", "Destination Country Name",
    "Origin Airport", "Destination Airport",
    "Distance (km)", "Passengers", "Avg. Total Fare(USD)"
}
if not expected_columns.issubset(df.columns):
    st.error(
        "CSV must contain columns: Origin Country Name, Destination Country Name, "
        "Origin Airport, Destination Airport, Distance (km), "
        "Passengers, Avg. Total Fare(USD)"
    )
    st.stop()

df = df.dropna()
df["Avg. Total Fare(USD)"] = df["Avg. Total Fare(USD)"].fillna(0)

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("📈 Policy & Economic Inputs")

ets_price = st.sidebar.slider(
    "Carbon price (EUR / tCO₂)",
    0, 400, 100, 5,
    help="Select the carbon price applied to aviation emissions.",
)

pass_through = (
    st.sidebar.slider(
        "Cost pass-through to fares (%)",
        0, 100, 80, 5,
        help="Share of carbon cost airlines embed in ticket prices.",
    ) / 100
)

global_gdp_growth = st.sidebar.slider(
    "Global Real GDP growth year-on-year (%)",
    -5.0, 8.0, 2.5, 0.1,
    help="Default GDP growth applied to all countries unless customized below.",
)

user_price_elasticity = st.sidebar.slider(
    "Demand price-elasticity (negative)",
    -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1,
    help="Passenger response to fare increases.",
)

user_gdp_elasticity = st.sidebar.slider(
    "Demand GDP-elasticity",
    0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1,
    help="Passenger response to GDP growth.",
)

# ----------------------
# Optional per-country GDP growth
# ----------------------
st.sidebar.markdown("### Optional: Adjust GDP Growth by Origin Country")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth for Specific Countries"):
    origin_countries = df["Origin Country Name"].dropna().unique()
    for country in sorted(origin_countries):
        gdp_growth_by_country[country] = st.slider(
            f"{country}",
            -5.0, 8.0, global_gdp_growth, 0.1,
            key=f"gdp_{country}",
        )

# ----------------------
# Carbon cost per passenger
# ----------------------
df["CO2 per pax (kg)"] = df["Distance (km)"] * EMISSION_FACTOR
df["Carbon cost per pax"] = (df["CO2 per pax (kg)"] / 1000) * ets_price * pass_through
df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + df["Carbon cost per pax"]
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

# ----------------------
# Elasticity model
# ----------------------
fare_factor = (
    (df["New Avg Fare"] / df["Avg. Total Fare(USD)"])
    .replace([np.inf, -np.inf], np.nan)
    ** user_price_elasticity
)
df["GDP Growth (%)"] = df["Origin Country"].map(gdp_growth_by_country).fillna(global_gdp_growth)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"] / 100) ** user_gdp_elasticity

df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# ----------------------
# Output
# ----------------------
st.subheader("📊 Passenger Simulation Results (Airport-Pair Level)")
st.dataframe(
    df[
        [
            "Origin Airport", "Destination Airport",
            "Origin Country Name", "Destination Country Name",
            "Passengers", "Distance (km)", "CO2 per pax (kg)",
            "Avg. Total Fare(USD)", "Carbon cost per pax",
            "New Avg Fare", "Passengers after policy", "Passenger Δ (%)"
        ]
    ],
    use_container_width=True,
)

# Aggregate by origin country
origin_summary = df.groupby("Origin Country", as_index=False).agg(
    {
        "Passengers": "sum",
        "Passengers after policy": "sum",
        "CO2 per pax (kg)": "mean",
    }
)
origin_summary["Relative Change (%)"] = (
    origin_summary["Passengers after policy"] / origin_summary["Passengers"] - 1
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
    total_passengers_2019 = df["Passengers"].sum()
    total_passengers_policy = df["Passengers after policy"].sum()
    passenger_delta = (total_passengers_policy / total_passengers_2019 - 1) * 100
    st.metric(
        "Total Passengers (m)",
        f"{total_passengers_policy / 1e6:,.2f} M",
        delta=f"{passenger_delta:+.1f}% vs 2019",
    )
with col2:
    avg_carbon_cost = df["Carbon cost per pax"].mean()
    st.metric("Avg Carbon Cost per Ticket", f"€{avg_carbon_cost:.2f}")

st.info(
    "💡 Each country inherits the global GDP growth unless adjusted manually in the sidebar dropdown."
)

st.caption("Data: Sabre MI (or dummy) · Visualization powered by Streamlit & Plotly")
