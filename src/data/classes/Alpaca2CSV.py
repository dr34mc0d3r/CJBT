"""download alpaca data for date / dates"""

import pandas as pd
from datetime import datetime
import os

class Alpaca2CSV:
    def __init__(self, json_data):
        self.json_data = json_data
        
    def process_to_df(self):
        # Extract the bars data and symbol
        symbol = list(self.json_data['bars'].keys())[0]
        bars_data = self.json_data['bars'][symbol]
        
        # Create initial DataFrame
        df = pd.DataFrame(bars_data)
        
        # Set 't' as DateTimeIndex and remove timezone
        df.index = pd.to_datetime(df['t'])
        df.index = df.index.tz_localize(None)
        
        # Get the target date from the first timestamp
        target_date = df.index[0].strftime('%Y-%m-%d')
        
        # Filter rows to only include those matching the target date
        df = df[df.index.strftime('%Y-%m-%d') == target_date]
        
        # Define time boundaries for the target date
        start_time = pd.to_datetime(f"{target_date} 14:30:00")
        end_time = pd.to_datetime(f"{target_date} 20:59:00")
        
        # Filter rows to only include times between 14:30:00 and 20:59:00
        df = df[(df.index >= start_time) & (df.index <= end_time)]
        
        # Rename columns
        df = df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume',
            'vw': 'VolumeWeighted'
        })
        
        # Select only the required columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'VolumeWeighted']]
        
        return df, symbol
    
    def save_to_csv(self, overwrite=True):
        """
        Save the DataFrame to a CSV file.
        
        Parameters:
        - overwrite (bool): If True, overwrite existing file; if False, skip if file exists.
        """
        # Process the data
        df, symbol = self.process_to_df()
        
        # Get date from first timestamp for filename
        date_str = df.index[0].strftime('%Y-%m-%d')
        
        # Create directory if it doesn't exist
        os.makedirs(f'/Users/chrisjackson/Desktop/DEV/python/data/1m/{symbol}', exist_ok=True)
        
        # Create filename
        filename = f'/Users/chrisjackson/Desktop/DEV/python/data/1m/{symbol}/{symbol}_{date_str}.csv'
        
        # Save to CSV, overwriting if file exists and overwrite=True
        if overwrite or not os.path.exists(filename):
            df.to_csv(filename)
        else:
            print(f"File {filename} already exists and overwrite is False; skipping.")
        
        return filename

# Example usage:
# if __name__ == "__main__":
#     sample_json = {
#         'bars': {
#             'AAPL': [
#                 {'c': 131, 'h': 131, 'l': 130.28, 'n': 208, 'o': 130.28, 
#                  't': '2024-01-03T14:30:00Z', 'v': 8174, 'vw': 130.880173},
#                 {'c': 131.1, 'h': 131.17, 'l': 130.87, 'n': 157, 'o': 130.87, 
#                  't': '2024-01-03T18:59:00Z', 'v': 8820, 'vw': 130.931663}
#             ]
#         },
#         'next_page_token': None
#     }
    
#     # Create instance and process
#     converter = Alpaca2CSV(sample_json)
#     filename = converter.save_to_csv()
#     print(f"Data saved to: {filename}")
    
#     # To see the DataFrame
#     df, _ = converter.process_to_df()
#     print("\nResulting DataFrame:")
#     print(df)