import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Model configuration
# ----------------------

CO2_KG_PER_PAX = 200  # average kg CO₂ per passenger
PRICE_ELASTICITY_DEMAND = -0.8
GDP_ELASTICITY_DEMAND = 1.4

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="Country-Level Aviation ETS Simulator", layout="wide")
st.title("🌍 JETPAS - Country-Level Aviation ETS Scenario Simulator")
st.markdown("Simulate how carbon pricing affects air travel between countries.")

with st.sidebar:
    st.header("📈 Policy & Economic Inputs")

    ets_price = st.slider(
        "Carbon price (EUR / tCO₂)", 0, 400, 100, 5,
        help="Select the carbon price applied to aviation emissions."
    )

    pass_through = st.slider(
        "Cost pass‑through to fares (%)", 0, 100, 80, 5,
        help="Share of carbon cost airlines embed in ticket prices."
    )

    gdp_growth = st.slider(
        "Real GDP growth year‑on‑year (%)", -5.0, 8.0, 2.5, 0.1,
        help="Macro driver for demand expansion."
    )

    user_price_elasticity = st.slider(
        "Demand price‑elasticity (negative)", -2.0, -0.1, PRICE_ELASTICITY_DEMAND, 0.1,
        help="Passenger response to fare increases."
    )

    user_gdp_elasticity = st.slider(
        "Demand GDP‑elasticity", 0.5, 2.0, GDP_ELASTICITY_DEMAND, 0.1,
        help="Passenger response to GDP growth."
    )

    uploaded_file = st.file_uploader("Upload country-pair passenger CSV", type=["csv"])

# ----------------------
# Processing uploaded file
# ----------------------

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    expected_columns = {"Origin_Country", "Destination_Country", "Passengers_2019", "Total_Fare_EUR"}
    if not expected_columns.issubset(df.columns):
        st.error("CSV must contain columns: Origin_Country, Destination_Country, Passengers_2019, Total_Fare_EUR")
    else:
        # Compute average fare per passenger
        df["Avg Fare (EUR)"] = df["Total_Fare_EUR"] / df["Passengers_2019"]
        df["Avg Fare (EUR)"] = df["Avg Fare (EUR)"].fillna(0)

        # Compute carbon cost per passenger
        carbon_cost = ets_price * (CO2_KG_PER_PAX / 1000) * (pass_through / 100)
        df["Carbon cost per pax"] = carbon_cost
        df["New Avg Fare"] = df["Avg Fare (EUR)"] + carbon_cost
        df["Fare Δ (%)"] = (df["New Avg Fare"] / df["Avg Fare (EUR)"] - 1) * 100

        # Elasticity-based demand model
        fare_change_factor = (df["New Avg Fare"] / df["Avg Fare (EUR)"]).replace([np.inf, -np.inf], np.nan) ** user_price_elasticity
        gdp_factor = (1 + gdp_growth / 100) ** user_gdp_elasticity

        df["Passengers after policy"] = df["Passengers_2019"] * fare_change_factor * gdp_factor
        df["Passenger Δ (%)"] = (df["Passengers after policy"] / df["Passengers_2019"] - 1) * 100

        # Show updated data
        st.subheader("📊 Passenger Simulation Results")
        st.dataframe(df, use_container_width=True)

        # Aggregate by Origin Country for chart
        origin_summary = df.groupby("Origin_Country", as_index=False)["Passengers after policy"].sum()

        fig = px.bar(
            origin_summary,
            x="Origin_Country",
            y="Passengers after policy",
            title="Passenger Volume by Origin Country (After Policy)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2 = st.columns(2)
        with col1:
            total_passengers_2019 = df["Passengers_2019"].sum()
            total_passengers_policy = df["Passengers after policy"].sum()
            passenger_delta = (total_passengers_policy / total_passengers_2019 - 1) * 100
            st.metric(
                "Total Passengers (m)",
                f"{total_passengers_policy / 1e6:,.2f} M",
                delta=f"{passenger_delta:+.1f}% vs 2019"
            )
        with col2:
            st.metric("Avg Carbon Cost per Ticket", f"€{carbon_cost:.2f}")

        st.info(
            "💡 **Insight**: Higher ETS prices increase fares, which reduces passenger demand based on elasticity. "
            "Economic growth partially offsets this by boosting demand."
        )
else:
    st.warning("📁 Please upload a CSV file with country-to-country passenger and fare data.")

st.caption("Model by OpenAI · Data: ICAO, IATA, IMF · Visualization powered by Streamlit & Plotly")
