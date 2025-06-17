# aviation_ets_dashboard.py
"""
Streamlit dashboard: Global Aviation ETS Scenario Simulator
----------------------------------------------------------
This app provides a simplified, Oxford‑Economics‑style macro model for the
aviation sector.  Users can explore how changes in carbon‑price policy (e.g.,
EU Emissions Trading System allowances) may cascade through ticket prices and
ultimately influence passenger demand worldwide.

DISCLAIMER
~~~~~~~~~~
The model is intentionally simple and uses stylised elasticities and baseline
figures for illustrative purposes.  For robust forecasting, you would replace
these with detailed market data, route‑level carbon intensities, airline
business‑models, and dynamic macro regressions.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Model configuration
# ----------------------

REGIONS = [
    "North America",
    "Europe",
    "Asia‑Pacific",
    "Latin America",
    "Middle East & Africa",
]

# Baseline 2019 passenger numbers (millions) — ICAO & IATA historical stats
BASE_PASSENGERS_M = {
    "North America": 930,
    "Europe": 1050,
    "Asia‑Pacific": 2120,
    "Latin America": 300,
    "Middle East & Africa": 320,
}

# Average one‑way fare per passenger (EUR)
BASE_FARE_EUR = {
    "North America": 320,
    "Europe": 270,
    "Asia‑Pacific": 210,
    "Latin America": 190,
    "Middle East & Africa": 230,
}

# Mean CO₂ per passenger per flight (kg) — global average narrow‑body + wide‑body mix
CO2_KG_PER_PAX = 200  # constant for simplicity

# Elasticities (global averages, literature survey)
PRICE_ELASTICITY_DEMAND = -0.8   # %Δ pax / %Δ real ticket price
GDP_ELASTICITY_DEMAND = 1.4      # %Δ pax / %Δ real GDP

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="Aviation ETS Scenario Simulator", layout="wide")
st.title("✈️ JETPAS - Joint Economic & Traffic Policy Aviation Simulation")
st.markdown(
    "Simulate how different policies and economic developments"
    " affect global air transport."
)

with st.sidebar:
    st.header("Policy & Macro Inputs")

    ets_price = st.slider(
        label="Carbon price (EUR / tCO₂)",
        min_value=0,
        max_value=400,
        value=100,
        step=5,
        help="Select the carbon price applied to aviation emissions.",
    )

    pass_through = st.slider(
        label="Cost pass‑through to fares (%)",
        min_value=0,
        max_value=100,
        value=80,
        step=5,
        help="Share of carbon cost airlines embed in ticket prices.",
    )

    gdp_growth = st.slider(
        label="Real GDP growth year‑on‑year (%)",
        min_value=-5.0,
        max_value=8.0,
        value=2.5,
        step=0.1,
        help="Macro driver for demand expansion (baseline 0%).",
    )

    user_price_elasticity = st.slider(
        label="Demand price‑elasticity (negative)",
        min_value=-2.0,
        max_value=-0.1,
        value=PRICE_ELASTICITY_DEMAND,
        step=0.1,
        help="Percentage change in passengers for a 1% change in real fares.",
    )

    user_gdp_elasticity = st.slider(
        label="Demand GDP‑elasticity",
        min_value=0.5,
        max_value=2.0,
        value=GDP_ELASTICITY_DEMAND,
        step=0.1,
        help="Percentage change in passengers for a 1% change in real GDP.",
    )

st.subheader("Baseline stats & scenario impact")

# Build baseline DataFrame
base_df = pd.DataFrame(
    {
        "Region": REGIONS,
        "Passengers (m, 2019)": [BASE_PASSENGERS_M[r] for r in REGIONS],
        "Avg Fare (EUR)": [BASE_FARE_EUR[r] for r in REGIONS],
    }
)

# Carbon cost computation (EUR per passenger)
carbon_cost = ets_price * (CO2_KG_PER_PAX / 1000) * (pass_through / 100)

# New fare and relative change
base_df["Carbon cost per pax"] = carbon_cost
base_df["New Avg Fare"] = base_df["Avg Fare (EUR)"] + carbon_cost
base_df["Fare Δ (%)"] = (base_df["New Avg Fare"] / base_df["Avg Fare (EUR)"] - 1) * 100

# Demand response factors
fare_change_factor = (base_df["New Avg Fare"] / base_df["Avg Fare (EUR)"]) ** user_price_elasticity

gdp_factor = (1 + gdp_growth / 100) ** user_gdp_elasticity

base_df["Passengers after policy (m)"] = base_df["Passengers (m, 2019)"] * fare_change_factor * gdp_factor
base_df["Passenger Δ (%)"] = (
    base_df["Passengers after policy (m)"] / base_df["Passengers (m, 2019)"] - 1
) * 100

# Display tables
st.dataframe(base_df.set_index("Region"), use_container_width=True)

# Chart passenger impact
fig = px.bar(
    base_df,
    x="Region",
    y="Passengers after policy (m)",
    color="Region",
    title="Passenger volumes under selected scenario",
)
st.plotly_chart(fig, use_container_width=True)

# Sensitivity summary
st.markdown("### Key takeaways")
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Worldwide passengers (m)",
        f"{base_df['Passengers after policy (m)'].sum():,.0f}",
        delta=f"{base_df['Passenger Δ (%)'].mean():+.1f}% vs 2019",
    )
with col2:
    st.metric(
        "Average carbon cost per ticket", f"€{carbon_cost:,.2f}", delta=None
    )

st.info(
    "**Interpretation guidance**  |  Increasing the ETS price raises airfare through the carbon cost pass‑through.  "
    "Given the negative demand price‑elasticity, higher fares depress passenger numbers.  GDP growth partially offsets this by lifting overall demand."
)

st.caption(
    "Model source code: github.com/your‑repo  ·  Data sources: ICAO, IATA, Eurocontrol, IMF, Oxford Economics benchmarks."
)
