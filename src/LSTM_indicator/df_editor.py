"""df_editor allows you to download alpaca stock data to csv files"""

import numpy as np
import pandas_ta as ta
import pandas as pd
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.events import Tap
from bokeh.transform import factor_cmap
import os
from datetime import datetime
import re

# Extend Panel with the Bokeh backend
pn.extension('bokeh')

class DataManager():
    def __init__(self):
        self.large_dataframe = pd.DataFrame()
        self.pickleFilePath = ""
        self.short_ema_period = 0
        self.long_ema_period = 0

    def stocks_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['t'] = pd.to_datetime(df['t'])
        df.set_index('t', inplace=True)
        return df
    
    def build_df_from_directory(self, root_dir, break_out_after=100000):

        # Regular expression to capture the date in YYYY-MM-DD format
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

        # Sort files by creation time to process them in chronological order
        files_with_date = []
        dataframes = []
        

        # Walk through the directory and process each file
        for dirname, _, filenames in os.walk(root_dir):
            for filename in filenames:
                # Search for a date pattern in the filename
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group()  # Extract the date string, e.g., "2020-12-31"
                    try:
                        # Convert the date string to a datetime object
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        file_path = os.path.join(dirname, filename)
                        files_with_date.append((file_date, file_path))
                    except ValueError:
                        print(f"Papa encountered an error parsing the date in file: {filename}")
                else:
                    print(f"Papa couldn't find a date in file: {filename}")

        # Sort the list of files by the extracted date
        files_with_date.sort()

        # Create a sorted list of file paths based on the date
        sorted_file_paths = [file_path for _, file_path in files_with_date]

        break_out_after_counter = 1
        for path in sorted_file_paths:
            try:
                if break_out_after_counter > break_out_after:
                    break

                print(f"Loading CSV for {path}")
                df = self.stocks_data(path)
                dataframes.append(df)
                break_out_after_counter += 1
            except pd.errors.EmptyDataError:
                print(f"Papa noticed that {path} is empty. Skipping this one.")
            except FileNotFoundError:
                print(f"Papa couldn't find the file: {path}. Skipping.")
            except Exception as e:
                print(f"Papa encountered an error with {path}: {e}. Skipping.")

        if dataframes:
            large_dataframe = pd.concat(dataframes, axis=0, ignore_index=False)
            print("Papa successfully combined all the data into one DataFrame!")
            print("Sorting the data...")
            large_dataframe = large_dataframe.sort_index()
        else:
            large_dataframe = pd.DataFrame()
            print("Papa found no data to combine. The Resulting DataFrame is empty.")

        # print(len(large_dataframe))
        # print(large_dataframe.columns)
        # print(large_dataframe.info())
        # print(large_dataframe.tail())

        return large_dataframe
    
    # New function to add the 100 EMA using pandas-ta
    def add_ema(self, df, period=100, title="long"):
        """
        Calculates the Exponential Moving Average (EMA) for the 'Close' column
        and adds it as a new column 'EMA_100' to the DataFrame.
        """
        df[f"EMA_{period}"] = ta.ema(df['Close'], length=period)

        if title == "long":
            print(f"Papa calculated the {title} {period} EMA!")
        else:
            print(f"Papa calculated the {title} {period} EMA!")

        return df
    
    def save_dataframe_as_pickle(self, df):
        """
        Save a pandas DataFrame to a pickle file.

        Parameters:
            df (pd.DataFrame): The DataFrame to save.
            file_path (str): The file path (including filename) where the pickle will be stored.
        """

        df.to_pickle(self.pickleFilePath)
        print(f"DataFrame saved to {self.pickleFilePath}")

    def load_dataframe_from_pickle(self, pickleFilePath):
        """
        Load a pandas DataFrame from a pickle file if it exists,
        with error trapping to handle any issues during loading.

        Parameters:
            file_path (str): Path to the pickle file.

        Returns:
            pd.DataFrame or None: The loaded DataFrame if successful, 
                                or None if the file doesn't exist or an error occurs.
        """

        if not os.path.exists(pickleFilePath):
            print(f"File '{pickleFilePath}' does not exist.")
            return None

        try:
            df = pd.read_pickle(pickleFilePath)
            print(f"DataFrame loaded successfully from '{pickleFilePath}'.")
            return df
        except Exception as e:
            print(f"Error loading pickle file '{pickleFilePath}': {e}")
            return None

        

class StockApp:
    """
    Reads stock data from a CSV file and sets the datetime column as the index.

    Parameters:
        csv_path (str): Path to the CSV file containing stock data.

    Returns:
        pd.DataFrame: DataFrame with the datetime index and stock data.
    
    Raises:
        FileNotFoundError: If the CSV file is not found.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """

    def __init__(self, df, datamanager):
        # Generate sample OHLCV data
        self.dataManager = datamanager
        self.df = df
        self.selected_date = None
        
        
        # Ensure 'Signal' column exists (defaulting to 0)
        if 'Signal' not in self.df.columns:
            self.df['Signal'] = 0
        
        # Create a ColumnDataSource for the "Close" price line
        self.source = ColumnDataSource(data={
            'date': self.df.index,
            'open': self.df['Open'],
            'high': self.df['High'],
            'low': self.df['Low'],
            'close': self.df['Close'],
            f"EMA_{self.dataManager.short_ema_period}": self.df[f"EMA_{self.dataManager.short_ema_period}"],
            f"EMA_{self.dataManager.long_ema_period}": self.df[f"EMA_{self.dataManager.long_ema_period}"]
        })
        
        # Create a ColumnDataSource for the "Signal" line (only non-zero signals)
        self.signal_source = ColumnDataSource(data=self.get_signal_data())
        
        # Build the Bokeh plot
        self.plot = self.create_plot()
        
        # Create interactive Panel widgets
        self.html_pane = pn.pane.HTML("<b>Click on the chart to select a date!</b>", width=400)
        self.text_input = pn.widgets.TextInput(name="Your Comment", placeholder="Type your comment here...")
        # Radio button options: 0 Hold, 1 Buy, 2 Sell (signal values to add)
        self.radio_button_group = pn.widgets.RadioButtonGroup(name="Signal Value", options=[0, 1, 2], button_type="success")
        self.submit_button = pn.widgets.Button(name="Submit", button_type="primary")
        self.submit_button.on_click(self.on_submit)

        self.save_button = pn.widgets.Button(name="Save", button_type="primary")
        self.save_button.on_click(self.on_save)
        
        # Compose the layout:
        # 1. The Bokeh chart on top.
        # 2. A row with the HTML pane showing the selected date.
        # 3. A row with text input, radio buttons, and submit button.
        self.layout = pn.Column(
            self.plot,
            pn.Row(self.html_pane),
            pn.Row(self.text_input, self.radio_button_group, self.submit_button, self.save_button)
        )


    def get_signal_data(self):
        """Return dictionary data for non-zero Signal values for plotting with an offset (Close + Signal)."""
        df_signal = self.df[self.df['Signal'] != 0]
        return {
            'date': list(df_signal.index),
            'offset': list(df_signal['Close']),
            # Convert signal values to strings
            'signal': list(df_signal['Signal'].astype(str))
        }

    def create_plot(self):
        """Create the Bokeh plot with OHLCV 'Close' line and the 'Signal' line."""
        p = figure(x_axis_type="datetime", title="OHLCV Data", height=400, width=800)

        # Plot the Close Price as a blue line
        p.line('date', 'close', source=self.source, line_width=2, color=(173, 173, 173), legend_label="Close Price")

        # Plot the df[f"EMA_{period}"]
        short_ema_period_column_name = f"EMA_{self.dataManager.short_ema_period}"
        p.line('date', short_ema_period_column_name, source=self.source, line_width=2, color=(235, 64, 52), legend_label=short_ema_period_column_name)

        long_ema_period_column_name = f"EMA_{self.dataManager.long_ema_period}"
        p.line('date', long_ema_period_column_name, source=self.source, line_width=2, color=(52, 110, 235), legend_label=long_ema_period_column_name)


        # Configure hover tool
        hover = HoverTool(
            tooltips=[
                ('Date', '@date{%F %T}'),
                ('Open', '@open{0.2f}'),
                ('High', '@high{0.2f}'),
                ('Low', '@low{0.2f}'),
                ('Close', '@close{0.2f}'),
                (short_ema_period_column_name, f'@{short_ema_period_column_name}{{0.000000f}}'),  # Explicit 6 zeros
                (long_ema_period_column_name, f'@{long_ema_period_column_name}{{0.000000f}}')
            ],
            formatters={
                '@date': 'datetime'
            },
            mode='vline'
        )

        # Add hover tool to plot
        p.add_tools(hover)
        
        # Plot dots on the chart for 1, 2 signals
        p.circle(
            'date', 'offset', source=self.signal_source, size=8,
            color=factor_cmap('signal', palette=['red', 'blue'], factors=["1", "2"]),
            name="signal_circles"
        )
        
        # Add an event listener for tap (click) events on the chart
        p.on_event(Tap, self.on_plot_click)
        
        p.legend.location = "top_left"
        return p

    def on_plot_click(self, event):
        """Update the HTML pane with the date corresponding to the clicked point."""
        # Convert event.x (in ms since epoch) to a pandas Timestamp
        clicked_dt = pd.to_datetime(event.x, unit='ms')
        # Find the closest date in our DataFrame index
        idx = self.df.index.get_indexer([clicked_dt], method='nearest')
        closest_date = self.df.index[idx[0]]
        self.selected_date = closest_date
        self.html_pane.object = f"<b>Selected Date:</b> {closest_date}"
        print(f"Chart clicked at: {closest_date}")  # Log to the Python console

    def on_submit(self, event):
        """Handle the Submit button click: update Signal and refresh the plot."""
        if self.selected_date is None:
            self.html_pane.object = "<b>Please click on the chart to select a date first!</b>"
            return
        
        text_value = self.text_input.value
        radio_value = self.radio_button_group.value  # This will be -1 or 1 based on selection
        
        # Print the selected date, text input, and radio button value to the console
        print(f"Submitted - Date: {self.selected_date}, Text: {text_value}, Radio: {radio_value}")
        
        # Update the Signal column by adding the radio value to the selected date's Signal
        if self.selected_date in self.df.index:
            self.df.loc[self.selected_date, 'Signal'] = radio_value
        else:
            print("Selected date not in DataFrame!")
        
        # Refresh the signal data source so that the plot updates accordingly
        self.signal_source.data = self.get_signal_data()
        
        # Optionally update the plot title to reflect changes
        self.plot.title.text = "OHLCV Data (Updated with Signal)"
        
        # Clear the text input for the next entry
        self.text_input.value = ""

    def on_save(self, event):
        """Handle the Save button click: Output final df to console for validation."""
        filtered_df = self.df[self.df['Signal'].isin([1, 2])]
        print(filtered_df)
        print(self.df)

        # Save dataframe to file
        self.dataManager.save_dataframe_as_pickle(self.df)



    def show(self):
        """Return the complete Panel layout for serving."""
        return self.layout


# Main execution block
if __name__ == '__main__':


    dataManager = DataManager()
    # add EMA indicators values to datadrame
    dataManager.long_ema_period = 100
    dataManager.short_ema_period = 13

    root_dir = '/Users/chrisjackson/Desktop/DEV/python/data/1m/TSLA'
    dataManager.pickleFilePath = '/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pickle'

    # set EMA Short and Long Periods for the EMA indicators
    short_ema_period, long_ema_period = [13, 100]

    df = dataManager.load_dataframe_from_pickle(dataManager.pickleFilePath)

    if df is not None:
        print(f"Loading data from File '{dataManager.pickleFilePath}'")

    else:
        print(f"File '{dataManager.pickleFilePath}' does not exist. Loading from CSV files")
        df = dataManager.build_df_from_directory(root_dir, 100)

        
        dataManager.add_ema(df, dataManager.short_ema_period, "short")
        dataManager.add_ema(df, dataManager.long_ema_period, "long")

    app = StockApp(df, dataManager)

    # Serve the app (use 'panel serve <script_name>.py' to run this script)
    pn.serve(app.show, show=True)
