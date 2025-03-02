import streamlit as st
import yfinance as yf
from datetime import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error


@st.cache_data
def fetch_data(symbol):
    """
    Fetch stock data from Yahoo Finance.

    :param symbol:      Stock symbol to download.
    :return df:         Stock historical data downloaded from Yahoo Finance.
    """
    # Fetch stock data from Yahoo Finance
    df = yf.download(symbol, "2000-01-01", end=datetime.today())  # , progress=False)

    # Drop MultiIndex
    df.columns = df.columns.droplevel(1)

    # Reset index
    df.reset_index(inplace=True)

    # Reformat date field
    df["Date"] = df["Date"].dt.date

    return df


def plot_candlestick_chart(symbol, stock_data):
    """
    Plotting a candlestick chart using Plotly.

    :param symbol:          Stock symbol.
    :param stock_data:      Historical stock data.
    :return:
    """

    # Plotting a candlestick chart using Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Candlestick(
        x=stock_data["Date"],
        open=stock_data["Open"],
        high=stock_data["High"],
        low=stock_data["Low"],
        close=stock_data["Close"]),
        secondary_y=True)

    fig.add_trace(go.Bar(x=stock_data["Date"], y=stock_data["Volume"]), secondary_y=False)

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
            dict(values=["2023-12-25", "2024-01-01"])  # hide Christmas and New Year's
        ]
    )

    # Customize layout
    fig.update_layout(
        title=f"{symbol} Stock Price Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig)

    return


def predict(model, num, data):
    num = int(num)

    if model == 'LinearRegression':
        engine = LinearRegression()
        model_engine(engine, num, data)
    elif model == 'RandomForestRegressor':
        engine = RandomForestRegressor()
        model_engine(engine, num, data)
    elif model == 'ExtraTreesRegressor':
        engine = ExtraTreesRegressor()
        model_engine(engine, num, data)
    elif model == 'KNeighborsRegressor':
        engine = KNeighborsRegressor()
        model_engine(engine, num, data)
    else:
        engine = XGBRegressor()
        model_engine(engine, num, data)

    return


def model_engine(model, num, data):
    scaler = StandardScaler()
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    # Spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # Training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

    return
