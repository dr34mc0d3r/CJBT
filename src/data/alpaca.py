"""
This module uses the Alpaca API to fetch historical stock data and save it as CSV files.
It processes data for a given date range and converts the API's JSON response into a CSV format.
"""


import requests
from datetime import datetime, timedelta
import time
import pytz
import pandas as pd

import sys
from dotenv import load_dotenv
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classes.Alpaca2CSV import Alpaca2CSV
from classes.DateRange import DayDateRange

# Load environment variables from the .env file (if present)
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')

def get_historical_data(symbol, start_date, end_date):
    """
    Retrieve historical market data for a given stock symbol from the Alpaca API.

    Parameters:
        symbol (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the data in ISO format (e.g., '2024-01-01').
        end_date (str): The end date for the data in ISO format (e.g., '2024-01-02').

    Returns:
        dict: A JSON dictionary containing historical bars data from Alpaca.
               The structure typically includes a 'bars' key with the stock data and possibly a 'next_page_token'.
    
    Note:
        Consider adding error handling for non-200 HTTP responses.
    """



    url = (
        f"https://data.alpaca.markets/v2/stocks/bars?"
        f"symbols={symbol}&"
        f"timeframe=1Min&"
        f"start={start_date}&"
        f"end={end_date}&"
        f"limit=1000&"
        f"adjustment=split&"
        f"feed=iex&"
        f"sort=asc"
    )
    
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
    response = requests.get(url, headers=headers)

    # print(url)
    out = response.json()
    

    return out
    # # return contains
    # {'bars': {'AAPL': [
    #     {'c': 131, 'h': 131, 'l': 130.28, 'n': 208, 'o': 130.28, 't': '2024-01-03T09:30:00Z', 'v': 8174, 'vw': 130.880173},
    #     {'c': 131.1, 'h': 131.17, 'l': 130.87, 'n': 157, 'o': 130.87, 't': '2024-01-03T16:00:00Z', 'v': 8820, 'vw': 130.931663}
    # ]}, 'next_page_token': None}


def get_date_range(enddate: str) -> tuple[datetime, datetime]:
    """
    Compute a date range ending on the provided date.

    Given an end date in 'YYYY-MM-DD' format, this function calculates and returns
    the start date (the previous day) and the end date as datetime objects.

    Parameters:
        enddate (str): The end date as a string in 'YYYY-MM-DD' format.

    Returns:
        tuple: A tuple containing two datetime objects (start_date, end_date).

    Raises:
        ValueError: If the input date string is not in the 'YYYY-MM-DD' format.
    """
    try:
        # Convert enddate string to datetime
        end_date = datetime.strptime(enddate, "%Y-%m-%d")
        
        # Calculate startdate as the previous day
        start_date = end_date - timedelta(days=1)
        
        # Convert back to strings in 'YYYY-MM-DD' format
        # startdate_str = start_date.strftime("%Y-%m-%d")
        # enddate_str = end_date.strftime("%Y-%m-%d")
        
        return (start_date, end_date)
    
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD'. {e}")
    




if __name__ == "__main__":
    """
    Main execution block:
    - Set the timezone for data processing.
    - Record the starting time to measure script execution duration.
    - Create a date range using DayDateRange.
    - For each date, calculate the appropriate start and end dates.
    - Fetch historical data using the Alpaca API.
    - Process and save the fetched data as a CSV file.
    - Output the elapsed time for the entire operation.
    """
    
    timezone = pytz.timezone('US/Eastern')
    # Record start time
    start_time = time.time()

    symbol = "TSLA"

    # Localize start and end dates to 'America/New_York' time zone
    start_date = pd.Timestamp('2020-12-31 09:30:00', tz='America/New_York')  # Market open time
    end_date = pd.Timestamp('2025-03-12 16:30:00', tz='America/New_York')    # 5 minutes later

    # Convert to strings in the desired format
    start_str = start_date.strftime("%Y-%m-%d") # %H:%M:%S %Z%z")
    end_str = end_date.strftime("%Y-%m-%d") # %H:%M:%S %Z%z")

    date_range = DayDateRange(start_str, end_str, True) # YYYY-MM-DD'
    dates = date_range.get_dates_between()
    print("Returned list:", len(dates))

    for date in dates:

        # Convert the datetime object to a string in 'YYYY-MM-DD' format.
        date_str = date.strftime("%Y-%m-%d")

        # Calculate the start and end dates for the API request based on the current date.
        startdateDT, enddateDT = get_date_range(date_str)

        # Convert datetime objects back to strings for API request formatting.
        startdateDT = startdateDT.strftime("%Y-%m-%d")
        enddateDT = enddateDT.strftime("%Y-%m-%d")

        # startdate_tz = timezone.localize(startdateDT)
        # startdate = startdate_tz.isoformat()

        # enddate_tz = timezone.localize(enddateDT)
        # enddate = startdate_tz.isoformat()

        # # url encode
        # startdate = startdate.replace(":", "%3A")
        # enddate = enddate.replace(":", "%3A")
        
        # Fetch historical data from Alpaca using the calculated date range.
        json_alpacaReturn = get_historical_data(symbol, startdateDT, enddateDT)

        # Check if the response contains the expected 'bars' data.
        if "bars" in json_alpacaReturn and symbol in json_alpacaReturn["bars"]:
            # print(len(json_alpacaReturn['bars'][f'{symbol}']))

            # Create instance and process
            converter = Alpaca2CSV(json_alpacaReturn)
            filename = converter.save_to_csv()
            print(f"Data saved to: {filename}")

            # To see the DataFrame
            # df, _ = converter.process_to_df()
            # print("\nResulting DataFrame:")
            # print(df)

        # Optional: Introduce a pause between requests to avoid rate limits.
        # time.sleep(2)

    # Record end time and compute the total elapsed time.
    end_time = time.time()

    # Calculate elapsed time in seconds
    elapsed_time = end_time - start_time

    # Convert elapsed time to hours, minutes, and seconds.
    hours = int(elapsed_time // 3600)  # Integer division by 3600 (seconds in an hour)
    minutes = int((elapsed_time % 3600) // 60)  # Remainder after hours, divided by 60 (seconds in a minute)
    seconds = elapsed_time % 60  # Remainder after minutes

    # Format the output
    time_str = []
    if hours > 0:
        time_str.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        time_str.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    time_str.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")

    # Display the formatted runtime.
    print(f"The loop took {', '.join(time_str)} to run.")
    
    