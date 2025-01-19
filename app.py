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

#     # Forecast for the next 7 trading days (excluding weekends)
#     future_days = []
#     last_date = df.index[-1]
#     while len(future_days) < 7:
#         last_date += pd.Timedelta(days=1)
#         if last_date.weekday() < 5:  # Exclude Saturdays (5) and Sundays (6)
#             future_days.append(last_date)

#     future_days_index = np.arange(len(df), len(df) + len(future_days)).reshape(-1, 1)
#     forecast = model.predict(future_days_index)

#     # Append forecasted data to DataFrame
#     forecast_df = pd.DataFrame({
#         'Date': future_days,
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
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Set page configuration
st.set_page_config(page_title="StockBuddy Assistant (NSE/BSE)", layout="wide", page_icon="📈")

# Fetch stock data using yfinance
def fetch_stock_data(symbol: str, period='6mo'):
    try:
        data = yf.download(symbol, period=period)
        if data.empty:
            raise ValueError("No data available for the given symbol.")
        
        # Exclude weekends (Saturday and Sunday)
        data = data[data.index.weekday < 5]  # Weekdays are 0-4 (Mon-Fri)
        
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Plot stock data based on selected price type
def plot_stock_data(df, price_type):
    fig, ax = plt.subplots(figsize=(12, 8))
    if price_type == "Open" or price_type == "All":
        ax.plot(df.index, df['Open'], label='Open Price', color='green', linestyle='--')
    if price_type == "High" or price_type == "All":
        ax.plot(df.index, df['High'], label='High Price', color='red', linestyle='--')
    if price_type == "Close" or price_type == "All":
        ax.plot(df.index, df['Close'], label='Closing Price', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'Stock Prices ({price_type})' if price_type != 'All' else 'Stock Prices (All Types)')
    ax.legend()
    st.pyplot(fig)

# Forecast stock prices using ARIMA
def forecast_stock_prices(df, days=7):
    df = df[['Close']]  # Use closing prices for ARIMA
    df = df.dropna()  # Drop any missing values

    # Fit ARIMA model
    model = ARIMA(df['Close'], order=(5, 1, 0))  # ARIMA(5,1,0)
    model_fit = model.fit()

    # Forecast for the next `days` trading days
    forecast = model_fit.forecast(steps=days)

    # Prepare future dates for the forecast (excluding weekends)
    last_date = df.index[-1]
    future_dates = []
    i = 1
    while len(future_dates) < days:
        future_candidate = last_date + pd.Timedelta(days=i)
        if future_candidate.weekday() < 5:  # Check if it's a weekday (0 to 4)
            future_dates.append(future_candidate)
        i += 1

    # Create a DataFrame for forecasted values
    forecast_df = pd.DataFrame({
        'Date': future_dates, 
        'Forecasted Price': forecast
    })

    # Plot combined data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df.index, df['Close'], label='Historical Prices', color='blue')
    ax.plot(forecast_df['Date'], forecast_df['Forecasted Price'], label='Forecasted Prices', color='red', linestyle='--', marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'Stock Prices with {days}-Day ARIMA Forecast')
    ax.legend()
    st.pyplot(fig)

    return forecast_df


# Decision-making based on forecast
def should_buy(forecast_df):
    prices = forecast_df['Forecasted Price']
    change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
    st.write(f"### Forecast Summary:")
    st.write(f"- **Starting Price:** {prices.iloc[0]:.2f}")
    st.write(f"- **Ending Price:** {prices.iloc[-1]:.2f}")
    st.write(f"- **Percentage Change:** {change:.2f}%")
    
    if change > 0:
        st.success(f"Forecast suggests an upward trend (+{change:.2f}%). It may be a good time to BUY the stock.")
    else:
        st.warning(f"Forecast suggests a downward trend ({change:.2f}%). It might not be the best time to buy the stock.")

# Main app
def main():
    st.title('📈 StockBuddy Assistant (NSE/BSE)')

    st.write("Enter stock symbols for NSE (e.g., RELIANCE.NS) or BSE (e.g., TCS.BO).")

    symbol = st.text_input('Enter Stock Symbol', 'RELIANCE.NS')
    forecast_days = st.slider("Select number of forecast days", min_value=1, max_value=30, value=7)
    price_type = st.selectbox("Select Price Type for Analysis", options=['Open', 'High', 'Low', 'Close', 'All'])

    if st.button('Get Data'):
        if symbol:
            df = fetch_stock_data(symbol)
            if df is not None:
                # Plot selected price type
                st.write(f"### {price_type} Price Chart")
                plot_stock_data(df, price_type)

                # Display the data table
                st.write("### Historical Stock Data")
                st.dataframe(df)

                # Show the forecast section
                st.write(f"### {forecast_days}-Day Stock Price ARIMA Forecast")
                forecast_df = forecast_stock_prices(df, days=forecast_days)

                # Display the forecasted data
                st.write("### Forecast Data")
                st.dataframe(forecast_df)

                # Provide recommendation
                st.write("### Recommendation")
                should_buy(forecast_df)

                # Provide download option
                st.write("### Download Data")
                combined_csv = pd.concat([df.reset_index(), forecast_df]).to_csv(index=False)
                st.download_button(label="Download Combined Data as CSV",
                                   data=combined_csv,
                                   file_name=f"{symbol}_stock_data.csv",
                                   mime="text/csv")
        else:
            st.warning("Please enter a stock symbol.")

if __name__ == "__main__":
    main()
