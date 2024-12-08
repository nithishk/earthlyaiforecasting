import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Add company logo and title
st.set_page_config(page_title="Steel Demand Forecasting", layout="wide")
st.image("a.png", width=150, caption="Powered by EarthlyAI")
st.title("Steel Demand Forecasting")

# Allow users to upload their own dataset
st.sidebar.title("Upload Steel Industry Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    steel_data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    steel_data.set_index("Date", inplace=True)
else:
    st.info("Using synthetic data as no file was uploaded.")
    # Generate synthetic dataset for the steel industry
    def generate_steel_data():
        np.random.seed(42)
        date_range = pd.date_range(start="2015-01-01", end="2023-12-31", freq="M")
        sales = np.random.normal(1000, 200, len(date_range)).clip(800, 1200)  # Sales in tons
        production = sales * np.random.uniform(1.1, 1.3, len(date_range))     # Production slightly exceeds sales
        co2_emissions = production * np.random.uniform(1.8, 2.2, len(date_range))  # CO2 emissions per ton produced

        steel_data = pd.DataFrame({
            "Date": date_range,
            "Sales (Tons)": sales,
            "Production (Tons)": production,
            "CO2 Emissions (Tons)": co2_emissions
        })
        return steel_data

    steel_data = generate_steel_data()
    steel_data.set_index("Date", inplace=True)

# Display dataset
st.subheader("Steel Industry Data")
st.write(steel_data)

# Visualize the data
st.subheader("Data Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steel_data.index, steel_data["Sales (Tons)"], label="Sales")
ax.plot(steel_data.index, steel_data["Production (Tons)"], label="Production")
ax.plot(steel_data.index, steel_data["CO2 Emissions (Tons)"], label="CO2 Emissions")
ax.set_title("Steel Industry Data")
ax.set_xlabel("Date")
ax.set_ylabel("Tons")
ax.legend()
ax.grid()
st.pyplot(fig)

# Forecasting Steel Demand
train = steel_data[:-24]
test = steel_data[-24:]

# Using Exponential Smoothing for Sales Forecasting
sales_model = ExponentialSmoothing(
    train["Sales (Tons)"], trend="add", seasonal="add", seasonal_periods=12
).fit()

# Forecast future sales
forecast_sales = sales_model.forecast(steps=len(test))

# Plot forecast vs actual
st.subheader("Steel Sales Forecast")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train.index, train["Sales (Tons)"], label="Training Data")
ax.plot(test.index, test["Sales (Tons)"], label="Actual Sales")
ax.plot(test.index, forecast_sales, label="Forecasted Sales", linestyle="--")
ax.set_title("Steel Sales Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Sales (Tons)")
ax.legend()
ax.grid()
st.pyplot(fig)

# Visualize CO2 emissions with forecasts
st.subheader("CO2 Emissions Forecast")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steel_data.index, steel_data["CO2 Emissions (Tons)"], label="Historical CO2 Emissions")
forecast_emissions = forecast_sales * 2.0  # Assuming average CO2 per ton is 2.0
ax.plot(test.index, forecast_emissions, label="Forecasted CO2 Emissions", linestyle="--")
ax.set_title("CO2 Emissions Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("CO2 Emissions (Tons)")
ax.legend()
ax.grid()
st.pyplot(fig)

# Display forecast vs actual
forecast_vs_actual = pd.DataFrame({
    "Actual Sales": test["Sales (Tons)"],
    "Forecasted Sales": forecast_sales
})

st.subheader("Forecast vs Actual")
st.write(forecast_vs_actual)

# Calculate forecast accuracy
mape = np.mean(np.abs((test["Sales (Tons)"] - forecast_sales) / test["Sales (Tons)"])) * 100
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Visualize sales and emissions together
st.subheader("Sales and Emissions Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test.index, test["Sales (Tons)"], label="Actual Sales")
ax.plot(test.index, forecast_sales, label="Forecasted Sales", linestyle="--")
ax.plot(test.index, forecast_emissions, label="Forecasted CO2 Emissions", linestyle="--")
ax.set_title("Sales and CO2 Emissions Comparison")
ax.set_xlabel("Date")
ax.set_ylabel("Tons")
ax.legend()
ax.grid()
st.pyplot(fig)
