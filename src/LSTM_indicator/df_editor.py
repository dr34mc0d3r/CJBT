"""df_editor allows you to download alpaca stock data to csv files"""

import numpy as np
import pandas_ta as ta
import pandas as pd
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.events import Tap
from bokeh.transform import factor_cmap
import os

# Extend Panel with the Bokeh backend
pn.extension('bokeh')

class DataManager():
    def __init__(self):
        self.data_manager_df = pd.DataFrame()

    def stocks_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['t'] = pd.to_datetime(df['t'])
        df.set_index('t', inplace=True)
        return df
    
    def build_df_from_directory(self, root_dir, break_out_after=100000):
        print("Papa training the model for forecasting...")
        files_with_ctime = []
        dataframes = []
        for dirname, _, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                try:
                    ctime = os.path.getctime(file_path)
                    files_with_ctime.append((ctime, file_path))
                except FileNotFoundError:
                    print(f"Papa couldn't find the file: {file_path}")
                except OSError as e:
                    print(f"Papa encountered an error with the file: {file_path}. Error: {e}")

        files_with_ctime.sort()
        sorted_file_paths = [file_path for _, file_path in files_with_ctime]

        break_out_after_counter = 0
        for path in sorted_file_paths:
            try:
                if break_out_after_counter > break_out_after:
                    break
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
    def add_ema(df, period=100):
        """
        Calculates the Exponential Moving Average (EMA) for the 'Close' column
        and adds it as a new column 'EMA_100' to the DataFrame.
        """
        df[f"EMA_{period}"] = ta.ema(df['Close'], length=period)
        print("Papa calculated the 100 EMA!")
        return df

        

class StockApp:
    def __init__(self, df):
        # Generate sample OHLCV data
        self.df = df
        self.selected_date = None
        
        # Ensure 'Signal' column exists (defaulting to 0)
        if 'Signal' not in self.df.columns:
            self.df['Signal'] = 0
        
        # Create a ColumnDataSource for the "Close" price line
        self.source = ColumnDataSource(data={
            'date': self.df.index,
            'close': self.df['Close']
        })
        
        # Create a ColumnDataSource for the "Signal" line (only non-zero signals)
        self.signal_source = ColumnDataSource(data=self.get_signal_data())
        
        # Build the Bokeh plot
        self.plot = self.create_plot()
        
        # Create interactive Panel widgets
        self.html_pane = pn.pane.HTML("<b>Click on the chart to select a date!</b>", width=400)
        self.text_input = pn.widgets.TextInput(name="Your Comment", placeholder="Type your comment here...")
        # Radio button options: -1 or 1 (signal values to add)
        self.radio_button_group = pn.widgets.RadioButtonGroup(name="Signal Value", options=[0, 1, 2], button_type="success")
        self.submit_button = pn.widgets.Button(name="Submit", button_type="primary")
        self.submit_button.on_click(self.on_submit)
        
        # Compose the layout:
        # 1. The Bokeh chart on top.
        # 2. A row with the HTML pane showing the selected date.
        # 3. A row with text input, radio buttons, and submit button.
        self.layout = pn.Column(
            self.plot,
            pn.Row(self.html_pane),
            pn.Row(self.text_input, self.radio_button_group, self.submit_button)
        )

    # def generate_data(self):
    #     """Generate sample OHLCV stock data with a datetime index."""
    #     dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    #     np.random.seed(42)  # For reproducibility
    #     # Create a random walk for prices
    #     price = np.cumsum(np.random.randn(100)) + 100
    #     # Generate OHLC values with some randomness
    #     high = price + np.abs(np.random.randn(100))
    #     low = price - np.abs(np.random.randn(100))
    #     open_price = price + np.random.randn(100)
    #     close = price + np.random.randn(100)
    #     volume = np.random.randint(1000, 5000, size=100)
        
    #     df = pd.DataFrame({
    #         'Open': open_price,
    #         'High': high,
    #         'Low': low,
    #         'Close': close,
    #         'Volume': volume
    #     }, index=dates)
    #     return df


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
        p.line('date', 'close', source=self.source, line_width=2, color="navy", legend_label="Close Price")
        
        # Plot the Signal values as a red line and circles (they will only show non-zero points)
        # p.line('date', 'signal', source=self.signal_source, line_width=2, color="red",
        #        legend_label="Signal", name="signal_line")
        # p.circle('date', 'offset', source=self.signal_source, size=8, color="red", name="signal_circles")
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
            self.df.loc[self.selected_date, 'Signal'] += radio_value
        else:
            print("Selected date not in DataFrame!")
        
        # Refresh the signal data source so that the plot updates accordingly
        self.signal_source.data = self.get_signal_data()
        
        # Optionally update the plot title to reflect changes
        self.plot.title.text = "OHLCV Data (Updated with Signal)"
        
        # Clear the text input for the next entry
        self.text_input.value = ""

    def show(self):
        """Return the complete Panel layout for serving."""
        return self.layout

# Instantiate the app

dataManager = DataManager()
root_dir = '/Users/chrisjackson/Desktop/DEV/python/data/1m/TSLA'
df = dataManager.build_df_from_directory(root_dir, 10)

# set EMA Short and Long Periods for the EMA indicators


app = StockApp(df)

# Serve the app (use 'panel serve <script_name>.py' to run this script)
pn.serve(app.show, show=True)
