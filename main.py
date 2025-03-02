import streamlit as st
import functions as fcn
import pandas as pd
import time

from prophet import Prophet
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Stock Price Forecasting", page_icon=":chart_with_upwards_trend:", layout="wide")

# Page title
st.title(":chart_with_upwards_trend: Stock Price Forecasting")

# Add search input textbox
symbol = st.sidebar.text_input("Search for Symbols:").upper()

try:
    # Download stock data from Yahoo Finance
    stock_data = fcn.fetch_data(symbol)

except ValueError:
    st.error("Please enter a symbol to start!")

else:

    if stock_data.shape[0] == 0:
        st.error("Symbol not found, please try again!")

    else:
        # Choose start date
        start_date = st.sidebar.date_input("Choose Start Date:", value=min(stock_data["Date"]),
                                           min_value=min(stock_data["Date"]), max_value=max(stock_data["Date"]))
        # Choose end date
        end_date = st.sidebar.date_input("Choose End Date:", value=max(stock_data["Date"]),
                                         min_value=start_date,max_value=max(stock_data["Date"]))

        # Apply date filter to stock data
        stock_data = stock_data[stock_data["Date"].between(start_date, end_date)]

        # Display number of data loaded
        st.sidebar.metric("Data Loaded (Days): ", stock_data.shape[0])

        # Show data
        if st.checkbox("Show Data"):
            st.write(stock_data)

        # Plot candlestick chart
        fcn.plot_candlestick_chart(symbol, stock_data)

        model_options = ["LinearRegression", "RandomForestRegressor", "ExtraTreesRegressor", "KNeighborsRegressor", "XGBoostRegressor"]
        model_selected = st.selectbox("Choose Model:", model_options)

        num = st.number_input('How many days forecast?', value=5)

        # Keep only the 'Date' and 'Close' columns for modeling
        data = stock_data[["Date", "Close"]]

        fcn.predict(model_selected, num, data)

        data.columns = ["ds", "y"]  # Prophet requires columns to be named 'ds' and 'y'

        # Reset index
        data.reset_index(drop=True, inplace=True)

        # Initialize and fit the Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(data)

        # Create a dataframe to hold predictions for the next 90 days
        future = model.make_future_dataframe(periods=90)

        # Generate predictions
        forecast = model.predict(future)

        # Plot the forecasted values
        model.plot(forecast)
        plt.title("Tesla Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Close Price (USD)")
        st.pyplot(plt)

        # Plot the forecast components (trend, weekly, yearly seasonality)
        model.plot_components(forecast)
        st.pyplot(plt)


