# aviation_ets_dashboard.py
"""
Streamlit dashboard: Global Aviation ETS Scenario Simulator
----------------------------------------------------------
A one-file dashboard you can **double-click** (or run via Python) and it will
self-launch Streamlit automatically.

Send it to colleagues: they only need Python 3.10+ and an internet browser.
When they open the file, the script detects whether it’s already inside a
Streamlit session; if not, it re-invokes `streamlit run` on itself and opens
in the default browser.

This version visualises **regional demand changes on a world map** instead of a
bar chart.
"""

from __future__ import annotations

import sys
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------
# Model configuration
# ----------------------

REGIONS = [
    "North America",
    "Europe",
    "Asia-Pacific",
    "Latin America",
    "Middle East & Africa",
]

BASE_PASSENGERS_M = {
    "North America": 930,
    "Europe": 1_050,
    "Asia-Pacific": 2_120,
    "Latin America": 300,
    "Middle East & Africa": 320,
}

BASE_FARE_EUR = {
    "North America": 320,
    "Europe": 270,
    "Asia-Pacific": 210,
    "Latin America": 190,
    "Middle East & Africa": 230,
}

# Approximate geographic centroids for world regions (lat, lon)
REGION_COORD = {
    "North America": (54, -105),          # central Canada/US
    "Europe": (54, 15),                   # Central Europe
    "Asia-Pacific": (10, 110),            # Southeast Asia / West Pacific
    "Latin America": (-15, -60),          # Brazil centroid
    "Middle East & Africa": (10, 25),     # Sudan/Sahel intersection
}

CO2_KG_PER_PAX = 200  # kg
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND = 1.4

# ----------------------
# Streamlit UI
# ----------------------

def render_dashboard() -> None:
    st.set_page_config(page_title="Aviation ETS Scenario Simulator", layout="wide")
    st.title("✈️ Aviation ETS Scenario Simulator")
    st.markdown(
        "Simulate how carbon-pricing cascades through airfares and passenger demand."
    )

    # ---------------- Sidebar controls ----------------
    with st.sidebar:
        st.header("Policy & Macro Inputs")
        ets_price = st.slider("Carbon price (EUR / tCO₂)", 0, 400, 100, 5)
        pass_through = st.slider("Cost pass-through to fares (%)", 0, 100, 80, 5)
        gdp_growth = st.slider("Real GDP growth YoY (%)", -5.0, 8.0, 2.5, 0.1)
        user_price_elasticity = st.slider(
            "Demand price-elasticity", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1
        )
        user_gdp_elasticity = st.slider(
            "Demand GDP-elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1
        )

    # --------------- Core calculations ---------------
    df = pd.DataFrame(
        {
            "Region": REGIONS,
            "Passengers_2019_m": [BASE_PASSENGERS_M[r] for r in REGIONS],
            "Avg_Fare_EUR": [BASE_FARE_EUR[r] for r in REGIONS],
        }
    )

    carbon_cost = ets_price * (CO2_KG_PER_PAX / 1000) * (pass_through / 100)
    df["Carbon_cost_pax"] = carbon_cost
    df["New_Avg_Fare"] = df["Avg_Fare_EUR"] + carbon_cost
    df["Fare_pct_delta"] = (df["New_Avg_Fare"] / df["Avg_Fare_EUR"] - 1) * 100

    fare_factor = (df["New_Avg_Fare"] / df["Avg_Fare_EUR"]) ** user_price_elasticity
    gdp_factor = (1 + gdp_growth / 100) ** user_gdp_elasticity
    df["Passengers_policy_m"] = df["Passengers_2019_m"] * fare_factor * gdp_factor
    df["Passengers_pct_delta"] = (df["Passengers_policy_m"] / df["Passengers_2019_m"] - 1) * 100

    # Add coordinates for geo-plot
    df["lat"] = df["Region"].apply(lambda r: REGION_COORD[r][0])
    df["lon"] = df["Region"].apply(lambda r: REGION_COORD[r][1])

    # ---------------- Data table ----------------
    st.subheader("Baseline stats & scenario impact")
    st.dataframe(
        df.set_index("Region")
          .rename(columns={
              "Passengers_2019_m": "Passengers (m, 2019)",
              "Avg_Fare_EUR": "Avg Fare (EUR)",
              "Passengers_policy_m": "Passengers after policy (m)",
              "Passengers_pct_delta": "Passenger Δ (%)",
          }),
        use_container_width=True,
    )

    # ---------------- Map visual ----------------
    st.markdown("### Passenger change by world region")
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        size="Passengers_policy_m",
        color="Passengers_pct_delta",
        hover_name="Region",
        hover_data={
            "Passengers_policy_m": ":,.0f",
            "Passengers_pct_delta": ":+.1f",
            "lat": False,
            "lon": False,
        },
        color_continuous_scale="RdYlGn",
        projection="natural earth",
        title="Passenger change under selected scenario (size = passengers, color = %Δ)",
    )
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Metrics ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Worldwide passengers (m)",
            f"{df['Passengers_policy_m'].sum():,.0f}",
            delta=f"{df['Passengers_pct_delta'].mean():+.1f}% vs 2019",
        )
    with col2:
        st.metric("Average carbon cost per ticket", f"€{carbon_cost:,.2f}")

    st.info(
        "Higher ETS prices increase fares (per pass-through). Demand falls with the chosen price-elasticity, while GDP growth can offset it. "
        "Bubble size shows absolute passenger volumes; color shows percentage change vs 2019."
    )

    st.caption("Data sources: ICAO, IATA, Eurocontrol, IMF. MIT-licensed.")

# ----------------------
# One-click launcher logic
# ----------------------

def _self_launch_streamlit_if_needed() -> None:
    """If executed via `python aviation_ets_dashboard.py`, relaunch under Streamlit."""
    if getattr(st.runtime, "exists", lambda: False)():
        return  # already inside Streamlit

    from streamlit.web import cli as stcli
    script_path = os.path.abspath(__file__)
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())


if __name__ == "__main__":
    _self_launch_streamlit_if_needed()
    render_dashboard()
