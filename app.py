# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import numpy as np

# # Set page configuration
# st.set_page_config(page_title="StockBuddy Assistant (NSE/BSE)",
#                    layout="wide",
#                    page_icon="📈")

# # Fetch stock data using yfinance
# def fetch_stock_data(symbol: str, period='6mo'):
#     data = yf.download(symbol, period=period)
#     return data

# # Plot stock data
# def plot_stock_data(df):
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.plot(df.index, df['Open'], label='Open Price', color='green', linestyle='--')
#     ax.plot(df.index, df['High'], label='High Price', color='red', linestyle='--')
#     ax.plot(df.index, df['Close'], label='Closing Price', color='orange')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price')
#     ax.set_title('Stock Prices (NSE/BSE)')
#     ax.legend()
#     st.pyplot(fig)

# # Forecast stock prices
# def forecast_stock_prices(df):
#     df['Days'] = np.arange(len(df))

#     # Use the last 60 days for forecasting
#     recent_data = df[-60:]

#     # Prepare the data for linear regression
#     X = recent_data[['Days']]
#     y = recent_data['Close']

#     # Train the model
#     model = LinearRegression()
#     model.fit(X, y)

#     # Forecast for the next 7 days
#     future_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
#     forecast = model.predict(future_days)

#     # Append forecasted data to DataFrame
#     forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
#     forecast_df = pd.DataFrame({
#         'Date': forecast_dates,
#         'Closing Price': forecast
#     })

#     # Combine historical and forecast data
#     combined_df = pd.concat([df[['Close']].reset_index(), forecast_df.rename(columns={'Date': 'index'})])

#     # Plot combined data
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.plot(df.index, df['Close'], label='Historical Prices', color='blue')
#     ax.plot(forecast_df['Date'], forecast_df['Closing Price'], label='Forecasted Prices', color='red', linestyle='--')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Closing Price')
#     ax.set_title('Stock Prices with 1-Week Forecast')
#     ax.legend()
#     st.pyplot(fig)

#     return forecast_df

# # Decision-making based on forecast
# def should_buy(forecast_df):
#     prices = forecast_df['Closing Price']
#     if prices.iloc[-1] > prices.iloc[0]:
#         st.success("Forecast suggests an upward trend! It may be a good time to BUY the stock.")
#     else:
#         st.warning("Forecast suggests a downward trend. It might not be the best time to buy the stock.")

# # Main app
# def main():
#     st.title('Stock Market Data Viewer with Forecast (NSE/BSE)')

#     st.write("Enter stock symbols for NSE (e.g., RELIANCE.NS) or BSE (e.g., TCS.BO).")
#     symbol = st.text_input('Enter Stock Symbol', 'RELIANCE.NS')
#     if st.button('Get Data'):
#         if symbol:
#             try:
#                 df = fetch_stock_data(symbol)

#                 # Show the stock price chart first
#                 st.write("### Stock Price Chart")
#                 plot_stock_data(df)

#                 # Display the data table
#                 st.write("### Historical Stock Data")
#                 st.dataframe(df)

#                 # Show the forecast section
#                 st.write("### 1-Week Stock Price Forecast")
#                 forecast_df = forecast_stock_prices(df)

#                 # Display the forecasted data
#                 st.write("### Forecast Data")
#                 st.dataframe(forecast_df)

#                 # Display recommendation based on forecast
#                 st.write("### Recommendation")
#                 should_buy(forecast_df)
#             except Exception as e:
#                 st.error(f"Error fetching data: {e}")
#         else:
#             st.warning("Please enter a stock symbol.")

# if __name__ == "__main__":
#     main()




import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page configuration
st.set_page_config(page_title="StockBuddy Assistant (NSE/BSE)",
                   layout="wide",
                   page_icon="📈")

# Fetch stock data using yfinance
def fetch_stock_data(symbol: str, period='6mo'):
    data = yf.download(symbol, period=period)
    return data

# Plot stock data
def plot_stock_data(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df.index, df['Open'], label='Open Price', color='green', linestyle='--')
    ax.plot(df.index, df['High'], label='High Price', color='red', linestyle='--')
    ax.plot(df.index, df['Close'], label='Closing Price', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock Prices (NSE/BSE)')
    ax.legend()
    st.pyplot(fig)

# Forecast stock prices
def forecast_stock_prices(df):
    df['Days'] = np.arange(len(df))

    # Use the last 60 days for forecasting
    recent_data = df[-60:]

    # Prepare the data for linear regression
    X = recent_data[['Days']]
    y = recent_data['Close']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast for the next 7 trading days (excluding weekends)
    future_days = []
    last_date = df.index[-1]
    while len(future_days) < 7:
        last_date += pd.Timedelta(days=1)
        if last_date.weekday() < 5:  # Exclude Saturdays (5) and Sundays (6)
            future_days.append(last_date)

    future_days_index = np.arange(len(df), len(df) + len(future_days)).reshape(-1, 1)
    forecast = model.predict(future_days_index)

    # Append forecasted data to DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_days,
        'Closing Price': forecast
    })

    # Combine historical and forecast data
    combined_df = pd.concat([df[['Close']].reset_index(), forecast_df.rename(columns={'Date': 'index'})])

    # Plot combined data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df.index, df['Close'], label='Historical Prices', color='blue')
    ax.plot(forecast_df['Date'], forecast_df['Closing Price'], label='Forecasted Prices', color='red', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Stock Prices with 1-Week Forecast')
    ax.legend()
    st.pyplot(fig)

    return forecast_df

# Decision-making based on forecast
def should_buy(forecast_df):
    prices = forecast_df['Closing Price']
    if prices.iloc[-1] > prices.iloc[0]:
        st.success("Forecast suggests an upward trend! It may be a good time to BUY the stock.")
    else:
        st.warning("Forecast suggests a downward trend. It might not be the best time to buy the stock.")

# Main app
def main():
    st.title('Stock Market Data Viewer with Forecast (NSE/BSE)')

    st.write("Enter stock symbols for NSE (e.g., RELIANCE.NS) or BSE (e.g., TCS.BO).")
    symbol = st.text_input('Enter Stock Symbol', 'RELIANCE.NS')
    if st.button('Get Data'):
        if symbol:
            try:
                df = fetch_stock_data(symbol)

                # Show the stock price chart first
                st.write("### Stock Price Chart")
                plot_stock_data(df)

                # Display the data table
                st.write("### Historical Stock Data")
                st.dataframe(df)

                # Show the forecast section
                st.write("### 1-Week Stock Price Forecast")
                forecast_df = forecast_stock_prices(df)

                # Display the forecasted data
                st.write("### Forecast Data")
                st.dataframe(forecast_df)

                # Display recommendation based on forecast
                st.write("### Recommendation")
                should_buy(forecast_df)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
        else:
            st.warning("Please enter a stock symbol.")

if __name__ == "__main__":
    main()
