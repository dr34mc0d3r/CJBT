

import os
import requests
import pandas as pd
from datetime import datetime
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = "https://data.alpaca.markets/v2"

# Headers for Alpaca API authentication
headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

# Define stock symbol and date range for validation
symbol = "TSLA"  # Example: Tesla stock
# Localize start and end dates to 'America/New_York' time zone
start_date = pd.Timestamp('2025-03-13 09:30:00', tz='America/New_York')  # Market open time
end_date = pd.Timestamp('2025-03-13 09:35:00', tz='America/New_York')    # 5 minutes later

# Convert to UTC for API requests
start_str = start_date.tz_convert('UTC').strftime("%Y-%m-%dT%H:%M:%SZ")
end_str = end_date.tz_convert('UTC').strftime("%Y-%m-%dT%H:%M:%SZ")

# Alpaca API request for 1-minute bars
params = {
    "symbols": symbol,
    "timeframe": "1Min",
    "start": start_str,
    "end": end_str,
    "limit": 1000  # Adjust if you need more bars
}

response = requests.get(f"{BASE_URL}/stocks/bars", headers=headers, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    bars = data.get("bars", {}).get(symbol, [])
    # Convert to DataFrame
    alpaca_df = pd.DataFrame(bars)
    if not alpaca_df.empty:
        alpaca_df['t'] = pd.to_datetime(alpaca_df['t'], utc=True)  # Ensure timestamps are UTC
        # Convert from UTC to 'America/New_York'
        alpaca_df['t'] = alpaca_df['t'].dt.tz_convert('America/New_York')
        alpaca_df = alpaca_df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })
        print("Alpaca Data:")
        print(alpaca_df[['timestamp', 'open', 'high', 'low', 'close']])
    else:
        print("Alpaca Data: No data returned for the specified range.")
else:
    print(f"Error fetching Alpaca data: {response.status_code} - {response.text}")
    alpaca_df = pd.DataFrame()  # Empty DataFrame if request fails

# Fetch 1-minute data from Yahoo Finance
yf_ticker = yf.Ticker(symbol)
yf_data = yf_ticker.history(start=start_date, end=end_date, interval='1m')

# Check if Yahoo Finance returned data
if not yf_data.empty:
    print("\nYahoo Finance Data:")
    print(yf_data)
    # Reset index and rename the timestamp column
    yf_data = yf_data.tz_convert('America/New_York')
    yf_data = yf_data.reset_index()
    yf_data = yf_data.rename(columns={'Datetime': 'timestamp'})
else:
    print("\nYahoo Finance Data: No data returned for the specified range.")
    yf_data = pd.DataFrame()

# Compare specific OHLC values for a given timestamp
specific_time = pd.Timestamp('2025-03-13 09:30:00', tz='America/New_York')  # Example timestamp to validate
alpaca_row = alpaca_df[alpaca_df['timestamp'] == specific_time] if not alpaca_df.empty else pd.DataFrame()
yf_row = yf_data[yf_data['timestamp'] == specific_time] if not yf_data.empty else pd.DataFrame()

if not alpaca_row.empty and not yf_row.empty:
    print(f"\nValidation for {specific_time}:")
    print(f"Alpaca OHLC: O={alpaca_row['open'].values[0]}, H={alpaca_row['high'].values[0]}, "
          f"L={alpaca_row['low'].values[0]}, C={alpaca_row['close'].values[0]}")
    print(f"Yahoo OHLC: O={yf_row['Open'].values[0]}, H={yf_row['High'].values[0]}, "
          f"L={yf_row['Low'].values[0]}, C={yf_row['Close'].values[0]}")
else:
    print(f"\nNo data found for {specific_time} in one or both sources.")
    if alpaca_row.empty:
        print("Alpaca data missing for this timestamp.")
    if yf_row.empty:
        print("Yahoo Finance data missing for this timestamp.")
