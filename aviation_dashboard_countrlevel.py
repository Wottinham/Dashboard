import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Model configuration
# ----------------------

CO2_KG_PER_PAX = 200
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND = 1.4

# ----------------------
# Helper – generate dummy data when no CSV is provided
# ----------------------
def load_dummy_data() -> pd.DataFrame:
    """Create a minimal, realistic country-pair dataset so users
    can try the simulator without uploading a file."""
    rng = np.random.default_rng(seed=42)
    origins = ["Germany", "France", "United States", "Japan"]
    destinations = ["Spain", "Italy", "United Kingdom", "Canada"]
    rows = []
    for o in origins:
        for d in destinations:
            if o == d:
                continue  # skip domestic legs here
            rows.append(
                {
                    "Origin Country": o,
                    "Destination Country": d,
                    "Passengers": int(rng.integers(50_000, 1_000_000)),
                    "Avg. Total Fare(USD)": round(rng.uniform(150, 700), 2),
                }
            )
    return pd.DataFrame(rows)

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="Country-Level Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Transport Policy Aviation Simulator")
st.markdown("Simulate air travel between countries.")

uploaded_file = st.sidebar.file_uploader(
    "Upload country-pair passenger CSV", type=["csv"]
)

# ----------------------
# Load data – CSV if provided, otherwise dummy
# ----------------------

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded.")
else:
    df = load_dummy_data()
    st.info(
        "🛈 No file uploaded – using **dummy data** so you can explore the simulator."
    )

# ----------------------
# Validate / clean data (works for both real & dummy)
# ----------------------

expected_columns = {
    "Origin Country",
    "Destination Country",
    "Passengers",
    "Avg. Total Fare(USD)",
}
if not expected_columns.issubset(df.columns):
    st.error(
        "CSV must contain columns: Origin Country, Destination Country, "
        "Passengers, Avg. Total Fare(USD)"
    )
    st.stop()

df = df.dropna()
df["Origin Country"] = df["Origin Country"].astype(str)
df = df[df["Origin Country"] != "00"]
df["Destination Country"] = df["Destination Country"].astype(str)
df = df[df["Destination Country"] != "00"]
df["Avg. Total Fare(USD)"] = df["Avg. Total Fare(USD)"].fillna(0)

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("📈 Policy & Economic Inputs")

ets_price = st.sidebar.slider(
    "Carbon price (EUR / tCO₂)",
    0,
    400,
    100,
    5,
    help="Select the carbon price applied to aviation emissions.",
)

pass_through = (
    st.sidebar.slider(
        "Cost pass-through to fares (%)",
        0,
        100,
        80,
        5,
        help="Share of carbon cost airlines embed in ticket prices.",
    )
    / 100
)

global_gdp_growth = st.sidebar.slider(
    "Global Real GDP growth year-on-year (%)",
    -5.0,
    8.0,
    2.5,
    0.1,
    help="Default GDP growth applied to all countries unless customized below.",
)

user_price_elasticity = st.sidebar.slider(
    "Demand price-elasticity (negative)",
    -2.0,
    -0.1,
    PRICE_ELASTICITY_DEMAND,
    0.1,
    help="Passenger response to fare increases.",
)

user_gdp_elasticity = st.sidebar.slider(
    "Demand GDP-elasticity",
    0.5,
    2.0,
    GDP_ELASTICITY_DEMAND,
    0.1,
    help="Passenger response to GDP growth.",
)

# ----------------------
# Optional per-country GDP growth
# ----------------------
st.sidebar.markdown("### Optional: Adjust GDP Growth by Origin Country")
gdp_growth_by_country = {}
with st.sidebar.expander("Customize GDP Growth for Specific Countries"):
    origin_countries = df["Origin Country"].dropna().unique()
    for country in sorted(origin_countries):
        gdp_growth_by_country[country] = st.slider(
            f"{country}",
            -5.0,
            8.0,
            global_gdp_growth,
            0.1,
            key=f"gdp_{country}",
        )

# ----------------------
# Carbon cost per passenger
# ----------------------
carbon_cost = ets_price * (CO2_KG_PER_PAX / 1000) * pass_through
df["Carbon cost per pax"] = carbon_cost
df["New Avg Fare"] = df["Avg. Total Fare(USD)"] + carbon_cost
df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg. Total Fare(USD)"] - 1) * 100

# ----------------------
# Elasticity model
# ----------------------
fare_factor = (
    (df["New Avg Fare"] / df["Avg. Total Fare(USD)"])
    .replace([np.inf, -np.inf], np.nan)
    ** user_price_elasticity
)
df["GDP Growth (%)"] = df["Origin Country"].map(gdp_growth_by_country).fillna(
    global_gdp_growth
)
df["GDP Growth Factor"] = (1 + df["GDP Growth (%)"] / 100) ** user_gdp_elasticity

df["Passengers after policy"] = df["Passengers"] * fare_factor * df["GDP Growth Factor"]
df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers"] - 1) * 100

# ----------------------
# Output
# ----------------------
st.subheader("📊 Passenger Simulation Results")
st.dataframe(
    df[
        [
            "Origin Country",
            "Destination Country",
            "Passengers",
            "Avg. Total Fare(USD)",
            "New Avg Fare",
            "Fare Δ (%)",
            "GDP Growth (%)",
            "Passengers after policy",
            "Passenger Δ (%)",
        ]
    ],
    use_container_width=True,
)

# Compute total base and policy passengers by origin country
origin_summary = df.groupby("Origin Country", as_index=False).agg(
    {"Passengers": "sum", "Passengers after policy": "sum"}
)

# Calculate relative change in %
origin_summary["Relative Change (%)"] = (
    (origin_summary["Passengers after policy"] / origin_summary["Passengers"]) - 1
) * 100

# Plot relative change
fig = px.bar(
    origin_summary,
    x="Origin Country",
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
    st.metric("Avg Carbon Cost per Ticket", f"€{carbon_cost:.2f}")

st.info(
    "💡 Each country inherits the global GDP growth unless adjusted manually in the sidebar dropdown."
)

st.caption("Data: Sabre MI (or dummy) · Visualization powered by Streamlit & Plotly")
