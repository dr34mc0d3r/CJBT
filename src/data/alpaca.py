
import requests
from datetime import datetime, timedelta
import time
import pytz

import sys
from dotenv import load_dotenv
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classes.Alpaca2CSV import Alpaca2CSV
from classes.DateRange import DayDateRange


API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')

def get_historical_data(symbol, start_date, end_date):

    url = f"https://data.alpaca.markets/v2/stocks/bars?" \
        f"symbols={symbol}&" \
        f"timeframe=1Min&" \
        f"start={start_date}&" \
        f"end={end_date}&" \
        f"limit=1000&" \
        f"adjustment=raw&" \
        f"feed=iex&" \
        f"sort=asc"
    
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
    response = requests.get(url, headers=headers)

    out = response.json()
    # print(url)

    return out
    # # return contains
    # {'bars': {'AAPL': [
    #     {'c': 131, 'h': 131, 'l': 130.28, 'n': 208, 'o': 130.28, 't': '2024-01-03T09:30:00Z', 'v': 8174, 'vw': 130.880173},
    #     {'c': 131.1, 'h': 131.17, 'l': 130.87, 'n': 157, 'o': 130.87, 't': '2024-01-03T16:00:00Z', 'v': 8820, 'vw': 130.931663}
    # ]}, 'next_page_token': None}


def get_date_range(enddate: str) -> tuple[str, str]:
    """
    Given an enddate, return startdate (previous day) and enddate as a tuple.
    
    Parameters:
    - enddate (str): End date in 'YYYY-MM-DD' format (e.g., '2023-01-02')
    
    Returns:
    - tuple: (startdate, enddate) as strings in 'YYYY-MM-DD' format
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
    
    timezone = pytz.timezone('US/Eastern')
    # Record start time
    start_time = time.time()

    symbol = "TSLA"

    date_range = DayDateRange("2021-01-01", "2025-03-05", True) # YYYY-MM-DD'
    dates = date_range.get_dates_between()
    print("Returned list:", len(dates))

    for date in dates:

        #to string
        date = date.strftime("%Y-%m-%d")
        startdateDT, enddateDT = get_date_range(date)

        startdateDT = startdateDT.strftime("%Y-%m-%d")
        enddateDT = enddateDT.strftime("%Y-%m-%d")

        # startdate_tz = timezone.localize(startdateDT)
        # startdate = startdate_tz.isoformat()

        # enddate_tz = timezone.localize(enddateDT)
        # enddate = startdate_tz.isoformat()

        # # url encode
        # startdate = startdate.replace(":", "%3A")
        # enddate = enddate.replace(":", "%3A")
        

        json_alpacaReturn = get_historical_data(symbol, startdateDT, enddateDT)

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

        # time.sleep(2)

    # Record end time
    end_time = time.time()

    # Calculate elapsed time in seconds
    elapsed_time = end_time - start_time

    # Convert to hours, minutes, and seconds
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

    print(f"The loop took {', '.join(time_str)} to run.")
    
    