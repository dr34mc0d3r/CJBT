"""df_editor allows you to download alpaca stock data to csv files"""

import os
from datetime import datetime
import re

# https://business-science.github.io/pytimetk/reference/
import pytimetk as tk

import numpy as np
import pandas_ta as ta
import pandas as pd
import panel as pn
from scipy.signal import butter, filtfilt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.events import Tap
from bokeh.transform import factor_cmap


# Extend Panel with the Bokeh backend
# pn.extension('bokeh')


import pandas as pd
import numpy as np

class TechnicalIndicators:

    @staticmethod
    # New function to add the 100 EMA using pandas-ta
    def add_ema(df, period=100, title="long"):
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
    
    @staticmethod
    def add_VWAP(df):
        """
        Add Volume Weighted Average Price (VWAP) to a pandas DataFrame with daily resets.
        
        Parameters:
            df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close', 'Volume' columns and DatetimeIndex.
        
        Returns:
            pd.DataFrame: DataFrame with an added 'VWAP' column.
        """
        # Ensure DataFrame has required columns
        required_columns = ['High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("DataFrame must contain 'High', 'Low', 'Close', and 'Volume' columns")
        
        # Use typical price as (High + Low + Close) / 3 for VWAP calculation
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Calculate Price * Volume
        df['Price_Volume'] = df['Typical_Price'] * df['Volume']
        
        # Extract date from DatetimeIndex for grouping
        df['vwapDate'] = df.index.date
        
        # Calculate cumulative sums within each day
        df['Cum_Price_Volume'] = df.groupby('vwapDate')['Price_Volume'].cumsum()
        df['Cum_Volume'] = df.groupby('vwapDate')['Volume'].cumsum()
        
        # Compute VWAP
        df['VWAP'] = df['Cum_Price_Volume'] / df['Cum_Volume']
        
        # Clean up temporary columns
        df.drop(columns=['Typical_Price', 'Price_Volume', 'Cum_Price_Volume', 'Cum_Volume', 'vwapDate'], inplace=True)
        
        return df
    
    @staticmethod
    def add_RSI(df, window=14):
        """Add Relative Strength Index (RSI) to the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame with 'Close' column.
            window (int): Number of periods for RSI calculation (default: 14).

        Returns:
            pd.DataFrame: DataFrame with added 'RSI' column.
        """
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df['RSI'] = rsi
        return df

    @staticmethod
    def add_Bollinger_Bands(df, window=20, num_std=2):
        """Add Bollinger Bands to the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame with 'Close' column.
            window (int): Number of periods for moving average (default: 20).
            num_std (float): Number of standard deviations for bands (default: 2).

        Returns:
            pd.DataFrame: DataFrame with added 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BBP' columns.
        """
        rolling_mean = df['Close'].rolling(window=window, min_periods=window).mean()
        rolling_std = df['Close'].rolling(window=window, min_periods=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        df['BB_Middle'] = rolling_mean
        df['BB_Upper'] = upper_band
        df['BB_Lower'] = lower_band
        df['BBP'] = (df['Close'] - lower_band) / (upper_band - lower_band)
        return df

    @staticmethod
    def add_ATR(df, window=14):
        """Add Average True Range (ATR) to the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns.
            window (int): Number of periods for ATR calculation (default: 14).

        Returns:
            pd.DataFrame: DataFrame with added 'ATR' column.
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window, min_periods=window).mean()
        df['ATR'] = atr
        return df

    @staticmethod
    def add_time_of_day(df, trading_minutes=390):
        """Add cyclical time-of-day encoding to the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame with DatetimeIndex.
            trading_minutes (int): Total trading minutes per day (default: 390, e.g., 9:30-16:00).

        Returns:
            pd.DataFrame: DataFrame with added 'Time_Sin' and 'Time_Cos' columns.
        """
        df['tdDate'] = df.index.date
        df['Minute_of_Day'] = df.groupby('tdDate').cumcount()
        normalized_time = df['Minute_of_Day'] / trading_minutes
        df['Time_Sin'] = np.sin(2 * np.pi * normalized_time)
        df['Time_Cos'] = np.cos(2 * np.pi * normalized_time)
        df.drop(['tdDate', 'Minute_of_Day'], axis=1, inplace=True)
        return df

    @staticmethod
    def add_returns(df):
        """Add minute-to-minute returns to the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame with 'Close' column.

        Returns:
            pd.DataFrame: DataFrame with added 'Returns' column.
        """
        df['Returns'] = df['Close'].pct_change()
        return df
    
    @staticmethod
    def add_lagged_returns(df, lags=[1, 2, 3, 4, 5]):
        """
        Add lagged returns from previous time steps to a pandas DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame with a 'Close' column.
            lags (list): List of integers representing the number of time steps to lag (default: [1, 2, 3]).
        
        Returns:
            pd.DataFrame: DataFrame with added columns 'Returns_Lag1', 'Returns_Lag2', etc.
        """
        # Ensure DataFrame has the required 'Close' column
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column")
        
        # Calculate minute-to-minute returns
        df['Returns'] = df['Close'].pct_change()
        
        # Add lagged returns for each specified lag
        for lag in lags:
            if lag <= 0:
                raise ValueError("Lags must be positive integers")
            df[f'Returns_Lag{lag}'] = df['Returns'].shift(lag)
        
        # Optionally, drop the 'Returns' column if you only want the lagged versions
        # df.drop(columns=['Returns'], inplace=True)  # Uncomment if desired
        
        return df
    
    @staticmethod
    def add_lagged_price_changes(df, lags=[1, 2, 3]):
        """
        Add lagged price changes (e.g., Close(t) - Close(t-1), High(t) - Low(t-1)) to a pandas DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame with 'Close', 'High', and 'Low' columns.
            lags (list): List of integers representing the number of time steps to lag (default: [1, 2, 3]).
        
        Returns:
            pd.DataFrame: DataFrame with added columns like 'Close_Diff_Lag1', 'High_Low_Diff_Lag1', etc.
        """
        # Ensure DataFrame has required columns
        required_columns = ['Close', 'High', 'Low']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("DataFrame must contain 'Close', 'High', and 'Low' columns")
        
        # Add lagged price changes for each specified lag
        for lag in lags:
            if lag <= 0:
                raise ValueError("Lags must be positive integers")
            
            # Close(t) - Close(t-lag): Change in closing price over lag periods
            df[f'Close_Diff_Lag{lag}'] = df['Close'] - df['Close'].shift(lag)
            
            # High(t) - Low(t-lag): Difference between current high and lagged low
            df[f'High_Low_Diff_Lag{lag}'] = df['High'] - df['Low'].shift(lag)
        
        return df
    
    @staticmethod
    def add_lagged_indicators(df, indicators, lags=[1, 2, 3]):
        """
        Add lagged values of specified indicators to a pandas DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the indicator columns.
            indicators (list): List of column names (e.g., ['EMA', 'RSI']) to lag.
            lags (list): List of integers representing the number of time steps to lag (default: [1, 2, 3]).
        
        Returns:
            pd.DataFrame: DataFrame with added columns like 'EMA_Lag1', 'RSI_Lag1', etc.
        """
        # Ensure DataFrame has the specified indicator columns
        missing_cols = [col for col in indicators if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing the following indicator columns: {missing_cols}")
        
        # Add lagged values for each indicator and lag
        for indicator in indicators:
            for lag in lags:
                if lag <= 0:
                    raise ValueError("Lags must be positive integers")
                # Create lagged column (e.g., 'EMA_Lag1')
                df[f'{indicator}_Lag{lag}'] = df[indicator].shift(lag)
        
        return df
    
    @staticmethod
    def add_roc(df, window=5):
        """
        Add Rate of Change (ROC) to a pandas DataFrame based on the Close price.
        
        Parameters:
            df (pd.DataFrame): DataFrame with a 'Close' column.
            window (int): Number of time steps over which to calculate ROC (default: 5).
        
        Returns:
            pd.DataFrame: DataFrame with an added 'ROC' column.
        """
        # Ensure DataFrame has the required 'Close' column
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column")
        
        if window <= 0:
            raise ValueError("Window must be a positive integer")
        
        # Calculate ROC: ((Close_t / Close_{t-window}) - 1) * 100
        df['ROC'] = ((df['Close'] / df['Close'].shift(window)) - 1) * 100
        
        return df
    
    @staticmethod
    def add_macd(df, fast_window=12, slow_window=26):
        """
        Add MACD (Moving Average Convergence Divergence) to a pandas DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame with a 'Close' column.
            fast_window (int): Period for the short-term EMA (default: 13).
            slow_window (int): Period for the long-term EMA (default: 100).
        
        Returns:
            pd.DataFrame: DataFrame with added 'EMA_short', 'EMA_long', and 'MACD' columns.
        """
        # Ensure DataFrame has the required 'Close' column
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column")
        
        if fast_window <= 0 or slow_window <= 0 or fast_window >= slow_window:
            raise ValueError("Windows must be positive integers, and fast_window must be less than slow_window")
        
        # Calculate short-term EMA
        df['MACD_EMA_short'] = df['Close'].ewm(span=fast_window, adjust=False).mean()
        
        # Calculate long-term EMA
        df['MACD_EMA_long'] = df['Close'].ewm(span=slow_window, adjust=False).mean()
        
        # Calculate MACD as the difference between short-term and long-term EMAs
        df['MACD'] = df['MACD_EMA_short'] - df['MACD_EMA_long']
        
        return df
    
    @staticmethod
    def add_volatility_volume_spikes(df, atr_window=14, volume_ma_window=5):
        """
        Add Volume Change and ATR Ratio to a pandas DataFrame to capture volatility and volume spikes.
        
        Parameters:
            df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close', and 'Volume' columns.
            atr_window (int): Period for ATR calculation (default: 14).
            volume_ma_window (int): Period for short-term volume moving average (default: 5).
        
        Returns:
            pd.DataFrame: DataFrame with added 'Volume_Change', 'Volume_MA', and 'ATR_Ratio' columns.
        """
        # Ensure DataFrame has required columns
        required_columns = ['High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("DataFrame must contain 'High', 'Low', 'Close', and 'Volume' columns")
        
        if atr_window <= 0 or volume_ma_window <= 0:
            raise ValueError("Window parameters must be positive integers")
        
        # Calculate Volume Change: Volume(t) - Volume(t-1)
        df['Volume_Change'] = df['Volume'].diff()
        
        # Calculate short-term Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=volume_ma_window, min_periods=1).mean()
        
        # Calculate ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=atr_window, min_periods=1).mean()
        
        # Calculate ATR Ratio: ATR / Close
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Optional: Drop temporary ATR column if not needed separately
        # df.drop(columns=['ATR'], inplace=True)  # Uncomment if desired
        
        return df
    
    @staticmethod
    def add_day_of_week(df):
        """
        Add Day of the Week as a feature to a pandas DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame with a DatetimeIndex.
        
        Returns:
            pd.DataFrame: DataFrame with an added 'DayOfWeek' column (0 = Monday, 6 = Sunday).
        """
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Extract day of the week (0 = Monday, 6 = Sunday)
        df['DayOfWeek'] = df.index.dayofweek
        
        return df

    @staticmethod
    def add_normalized_differences(df, EMA_ColumnName):
        """Add normalized price differences to the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame with 'Close', EMA_ColumnName, 'VWAP', 'ATR' columns.

        Returns:
            pd.DataFrame: DataFrame with added 'Norm_Diff_EMA' and 'Norm_Diff_VWAP' columns.
        """
        df['Norm_Diff_EMA'] = (df['Close'] - df[f"{EMA_ColumnName}"]) / df['ATR']
        df['Norm_Diff_VWAP'] = (df['Close'] - df['VWAP']) / df['ATR']
        return df
    
    @staticmethod
    # Butterworth filter functions
    def butter_lowpass(close_prices, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b_low, a_low = butter(order, normal_cutoff, btype='low', analog=False)

        lowpass_filtered = filtfilt(b_low, a_low, close_prices)
        df['Lowpass'] = lowpass_filtered
        return df

    def butter_highpass(close_prices, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b_high, a_high = butter(order, normal_cutoff, btype='high', analog=False)

        highpass_filtered = filtfilt(b_high, a_high, close_prices)
        df['Highpass'] = highpass_filtered + close_prices
        return df

    @staticmethod
    def add_all_features(df, rsi_window=14, bb_window=20, bb_std=2, atr_window=14, trading_minutes=390):
        """Add all technical indicators to the DataFrame in one call.

        Parameters:
            df (pd.DataFrame): DataFrame with required columns and DatetimeIndex.
            rsi_window (int): Window for RSI (default: 14).
            bb_window (int): Window for Bollinger Bands (default: 20).
            bb_std (float): Standard deviations for Bollinger Bands (default: 2).
            atr_window (int): Window for ATR (default: 14).
            trading_minutes (int): Trading minutes per day (default: 390).

        Returns:
            pd.DataFrame: DataFrame with all new columns added.
        """
        df = TechnicalIndicators.add_VWAP(df)
        df = TechnicalIndicators.add_RSI(df, window=rsi_window)
        df = TechnicalIndicators.add_Bollinger_Bands(df, window=bb_window, num_std=bb_std)
        df = TechnicalIndicators.add_ATR(df, window=atr_window)
        df = TechnicalIndicators.add_time_of_day(df, trading_minutes=trading_minutes)
        df = TechnicalIndicators.add_returns(df)
        df = TechnicalIndicators.add_lagged_returns(df, lags=[1, 2, 3, 4, 5])
        df = TechnicalIndicators.add_lagged_price_changes(df, lags=[1, 2, 3, 4, 5])
        df = TechnicalIndicators.add_roc(df, window=5)
        df = TechnicalIndicators.add_macd(df, fast_window=12, slow_window=26)
        df = TechnicalIndicators.add_volatility_volume_spikes(df, atr_window=14, volume_ma_window=5)
        df = TechnicalIndicators.add_day_of_week(df)

        close_prices = df['Close'].values
        fs = 1  # 1 sample per minute
        cutoff_freq = 0.01  # Adjust this based on your needs

        

        df = TechnicalIndicators.butter_highpass(close_prices, cutoff_freq, fs, order=5)

        df = TechnicalIndicators.butter_lowpass(close_prices, cutoff_freq, fs, order=5)

        return df

class DataManager():
    def __init__(self):
        self.large_dataframe = pd.DataFrame()
        self.pickleFilePath = ""
        # self.short_ema_period = 0
        # self.long_ema_period = 0

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

                # print(f"Loading CSV for {path}")
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
        
        
        # Ensure 'BuySignal' column exists (defaulting to 0)
        if 'BuySignal' not in self.df.columns:
            self.df['BuySignal'] = 0

        # Ensure 'SellSignal' column exists (defaulting to 0)
        if 'SellSignal' not in self.df.columns:
            self.df['SellSignal'] = 0
        
        # Create a ColumnDataSource for the "Close" price line
        self.source = ColumnDataSource(data={
            'date': self.df.index,
            'open': self.df['Open'],
            'high': self.df['High'],
            f"High_Low_Diff_Lag1": self.df["High_Low_Diff_Lag1"],
            f"High_Low_Diff_Lag2": self.df["High_Low_Diff_Lag2"],
            f"High_Low_Diff_Lag3": self.df["High_Low_Diff_Lag3"],
            f"High_Low_Diff_Lag4": self.df["High_Low_Diff_Lag4"],
            f"High_Low_Diff_Lag5": self.df["High_Low_Diff_Lag5"],
            'low': self.df['Low'],
            'close': self.df['Close'],
            f"Close_Diff_Lag1": self.df["Close_Diff_Lag1"],
            f"Close_Diff_Lag2": self.df["Close_Diff_Lag2"],
            f"Close_Diff_Lag3": self.df["Close_Diff_Lag3"],
            f"Close_Diff_Lag4": self.df["Close_Diff_Lag4"],
            f"Close_Diff_Lag5": self.df["Close_Diff_Lag5"],
            
            # f"EMA_{self.dataManager.short_ema_period}": self.df[f"EMA_{self.dataManager.short_ema_period}"],
            # f"EMA_{self.dataManager.long_ema_period}": self.df[f"EMA_{self.dataManager.long_ema_period}"],
            f"VWAP": self.df["VWAP"],
            f"BB_Upper": self.df["BB_Upper"],
            f"BB_Middle": self.df["BB_Middle"],
            f"BB_Lower": self.df["BB_Lower"],
            f"BBP": self.df["BBP"],
            f"RSI": self.df["RSI"],
            f"ATR": self.df["ATR"],
            f"Time_Sin": self.df["Time_Sin"],
            f"Time_Cos": self.df["Time_Cos"],
            f"Returns": self.df["Returns"],
            f"Returns_Lag1": self.df["Returns_Lag1"],
            f"Returns_Lag2": self.df["Returns_Lag2"],
            f"Returns_Lag3": self.df["Returns_Lag3"],
            f"Returns_Lag4": self.df["Returns_Lag4"],
            f"Returns_Lag5": self.df["Returns_Lag5"],
            f"ROC": self.df["ROC"],
            f"Volume_Change": self.df["Volume_Change"],
            f"Volume_MA": self.df["Volume_MA"],
            f"ATR_Ratio": self.df["ATR_Ratio"],
            f"DayOfWeek": self.df["DayOfWeek"],
            f"Lowpass": self.df["Lowpass"],
            f"Highpass": self.df["Highpass"],
            
            # f"Norm_Diff_EMA": self.df["Norm_Diff_EMA"],
            # f"Norm_Diff_VWAP": self.df["Norm_Diff_VWAP"]
        })

        
        # Create a ColumnDataSource for the "BuySignal" line
        self.buy_signal_source = ColumnDataSource(data=self.Buy_get_signal_data())

        # Create a ColumnDataSource for the "SellSignal" line
        self.sell_signal_source = ColumnDataSource(data=self.Sell_get_signal_data())
        
        # Build the Bokeh plot
        self.plot = self.create_plot()
        
        # Create interactive Panel widgets
        self.html_pane = pn.pane.HTML("<b>Click on the chart to select a date!</b>", width=400)

        # Radio button options: 0 NotBuy, 1 Buy (buy signal values to add)
        self.Buy_radio_button_group = pn.widgets.RadioButtonGroup(name="Buy_Signal Value", options=[0, 1], button_type="success")
        self.Buy_submit_button = pn.widgets.Button(name="Buy_Submit", button_type="primary")
        self.Buy_submit_button.on_click(self.Buy_on_submit)

        self.Buy_save_button = pn.widgets.Button(name="Buy_Save", button_type="primary")
        self.Buy_save_button.on_click(self.Buy_on_save)

        # Radio button options: 0 NotSell, 1 Sell (buy signal values to add)
        self.Sell_radio_button_group = pn.widgets.RadioButtonGroup(name="Sell_Signal Value", options=[0, 1], button_type="success")
        self.Sell_submit_button = pn.widgets.Button(name="Sell_Submit", button_type="primary")
        self.Sell_submit_button.on_click(self.Sell_on_submit)

        self.Sell_save_button = pn.widgets.Button(name="Sell_Save", button_type="primary")
        self.Sell_save_button.on_click(self.Sell_on_save)
        
        # Compose the layout:
        # 1. The Bokeh chart on top.
        # 2. A row with the HTML pane showing the selected date.
        # 3. A row with text input, radio buttons, and submit button.
        self.layout = pn.Column(
            self.plot,
            pn.Row(self.html_pane),
            pn.Row(self.Buy_radio_button_group, self.Buy_submit_button, self.Buy_save_button),
            pn.Row(self.Sell_radio_button_group, self.Sell_submit_button, self.Sell_save_button)
        )

    def get_color(self, colorIndex, background=True):
        # List of pastel colors for black background
        black_bg_colors = [
            (173, 216, 230),  # Light pastel blue
            (240, 182, 187),  # Soft pastel pink
            (144, 238, 144),  # Pale pastel green
            (255, 218, 185),  # Peach pastel
            (221, 160, 221),  # Pastel plum
            (176, 224, 230),  # Powder blue pastel
            (245, 245, 220),  # Creamy beige pastel
            (152, 251, 152),  # Mint pastel green
            (255, 192, 203),  # Baby pink pastel
            (230, 230, 250),  # Lavender pastel
            (173, 255, 47),   # Pastel lime green
            (240, 230, 140),  # Pale pastel yellow
            (175, 238, 238),  # Turquoise pastel
            (255, 228, 225),  # Misty rose pastel
            (147, 197, 114)   # Sage pastel green
        ]
        
        # List of slightly darker colors for white background
        white_bg_colors = [
            (70, 130, 180),   # Steel blue
            (220, 80, 100),   # Soft red
            (60, 179, 113),   # Medium sea green
            (255, 165, 79),   # Orange
            (186, 85, 211),   # Medium orchid
            (100, 149, 237),  # Cornflower blue
            (189, 183, 107),  # Dark khaki
            (50, 205, 50),    # Lime green
            (255, 105, 180),  # Hot pink
            (147, 112, 219),  # Medium purple
            (154, 205, 50),   # Yellow-green
            (218, 165, 32),   # Goldenrod
            (64, 224, 208),   # Turquoise
            (255, 99, 71),    # Tomato
            (107, 142, 35)    # Olive drab
        ]
        
        # Select the appropriate color list based on background
        if background:  # True for black background
            colors = black_bg_colors
        else:  # False for white background
            colors = white_bg_colors
        
        # Return the color at the specified index, wrapping around if index exceeds list length
        return colors[colorIndex % len(colors)]


    def Buy_get_signal_data(self):
        """Return dictionary data for non-zero BuySignal values for plotting with an offset (Close + Signal)."""
        df_signal = self.df[self.df['BuySignal'] != 0]
        return {
            'date': list(df_signal.index),
            'offset': list(df_signal['Close']),
            # Convert signal values to strings
            'buy_signal': list(df_signal['BuySignal'].astype(str))
        }
    
    def Sell_get_signal_data(self):
        """Return dictionary data for non-zero SellSignal values for plotting with an offset (Close + Signal)."""
        df_signal = self.df[self.df['SellSignal'] != 0]
        return {
            'date': list(df_signal.index),
            'offset': list(df_signal['Close']),
            # Convert signal values to strings
            'sell_signal': list(df_signal['SellSignal'].astype(str))
        }

    def create_plot(self):
        """Create the Bokeh plot with OHLCV 'Close' line and the 'Signal' line."""
        p = figure(x_axis_type="datetime", title="OHLCV Data", height=800, width=1200, sizing_mode="stretch_both")


        # Plot dots on the chart for 1 buy signals
        p.scatter(
            x='date', 
            y='offset', 
            source=self.buy_signal_source, 
            size=20, 
            color=(59, 161, 8), 
            marker="circle",  # Explicitly set marker type to circle
            name="buy_signal_circles"
        )

        # Plot dots on the chart for 1 sell signals
        p.scatter(
            x='date', 
            y='offset', 
            source=self.sell_signal_source, 
            size=20, 
            color=(161, 8, 130), 
            marker="circle",  # Explicitly set marker type to circle
            name="sell_signal_circles"
        )


        # Plot the Close Price as a blue line
        p.line('date', 'close', source=self.source, line_width=2, color=(0, 0, 0), visible=True, legend_label="Close Price")

        # # Plot the df[f"EMA_{period}"]
        # short_ema_period_column_name = f"EMA_{self.dataManager.short_ema_period}"
        # p.line('date', short_ema_period_column_name, source=self.source, line_width=2, color=(235, 64, 52), legend_label=short_ema_period_column_name)

        # long_ema_period_column_name = f"EMA_{self.dataManager.long_ema_period}"
        # p.line('date', long_ema_period_column_name, source=self.source, line_width=2, color=(52, 110, 235), legend_label=long_ema_period_column_name)


        # Plot the VWAP
        p.line('date', 'VWAP', source=self.source, line_width=2, color=self.get_color(7, False), visible=False, legend_label="VWAP")
        # Plot the BB_Upper
        p.line('date', 'BB_Upper', source=self.source, line_width=2, color=self.get_color(8, False), visible=True, legend_label="BB_Upper")
        # Plot the BB_Middle
        p.line('date', 'BB_Middle', source=self.source, line_width=2, color=self.get_color(9, False), visible=True, legend_label="BB_Middle")
        # Plot the BB_Lower
        p.line('date', 'BB_Lower', source=self.source, line_width=2, color=self.get_color(10, False), visible=True, legend_label="BB_Lower")
        # Plot the BBP
        p.line('date', 'BBP', source=self.source, line_width=2, color=self.get_color(11, False), visible=False, legend_label="BBP")

        # Plot the RSI
        p.line('date', 'RSI', source=self.source, line_width=2, color=self.get_color(0, False), visible=False, legend_label="RSI")
        # Plot the ATR
        p.line('date', 'ATR', source=self.source, line_width=2, color=self.get_color(1, False), visible=False, legend_label="ATR")
        # Plot the Time_Sin
        p.line('date', 'Time_Sin', source=self.source, line_width=2, color=self.get_color(2, False), visible=False, legend_label="Time_Sin")
        # Plot the Time_Cos
        p.line('date', 'Time_Cos', source=self.source, line_width=2, color=self.get_color(3, False), visible=False, legend_label="Time_Cos")
        # Plot the Returns
        p.line('date', 'Returns', source=self.source, line_width=2, color=self.get_color(4, False), visible=False, legend_label="Returns")
        # # Plot the Norm_Diff_EMA
        # p.line('date', 'Norm_Diff_EMA', source=self.source, line_width=2, color=self.get_color(5, False), visible=False, legend_label="Norm_Diff_EMA")
        # # Plot the Norm_Diff_VWAP
        # p.line('date', 'Norm_Diff_VWAP', source=self.source, line_width=2, color=self.get_color(6, False), visible=False, legend_label="Norm_Diff_VWAP")



        # Plot the High_Low_Diff_Lag1
        p.line('date', 'High_Low_Diff_Lag1', source=self.source, line_width=2, color=self.get_color(5, False), visible=False, legend_label="High_Low_Diff_Lag1")
        # Plot the High_Low_Diff_Lag2
        p.line('date', 'High_Low_Diff_Lag2', source=self.source, line_width=2, color=self.get_color(6, False), visible=False, legend_label="High_Low_Diff_Lag2")
        # Plot the High_Low_Diff_Lag3
        p.line('date', 'High_Low_Diff_Lag3', source=self.source, line_width=2, color=self.get_color(7, False), visible=False, legend_label="High_Low_Diff_Lag3")
        # Plot the High_Low_Diff_Lag4
        p.line('date', 'High_Low_Diff_Lag4', source=self.source, line_width=2, color=self.get_color(8, False), visible=False, legend_label="High_Low_Diff_Lag4")
        # Plot the High_Low_Diff_Lag5
        p.line('date', 'High_Low_Diff_Lag5', source=self.source, line_width=2, color=self.get_color(9, False), visible=False, legend_label="High_Low_Diff_Lag5")

        # Plot the Close_Diff_Lag1
        p.line('date', 'Close_Diff_Lag1', source=self.source, line_width=2, color=self.get_color(5, False), visible=False, legend_label="Close_Diff_Lag1")
        # Plot the Close_Diff_Lag2
        p.line('date', 'Close_Diff_Lag2', source=self.source, line_width=2, color=self.get_color(6, False), visible=False, legend_label="Close_Diff_Lag2")
        # Plot the Close_Diff_Lag3
        p.line('date', 'Close_Diff_Lag3', source=self.source, line_width=2, color=self.get_color(7, False), visible=False, legend_label="Close_Diff_Lag3")
        # Plot the Close_Diff_Lag4
        p.line('date', 'Close_Diff_Lag4', source=self.source, line_width=2, color=self.get_color(8, False), visible=False, legend_label="Close_Diff_Lag4")
        # Plot the Close_Diff_Lag5
        p.line('date', 'Close_Diff_Lag5', source=self.source, line_width=2, color=self.get_color(9, False), visible=False, legend_label="Close_Diff_Lag5")

        # Plot the Returns_Lag1
        p.line('date', 'Returns_Lag1', source=self.source, line_width=2, color=self.get_color(5, False), visible=False, legend_label="Returns_Lag1")
        # Plot the Returns_Lag2
        p.line('date', 'Returns_Lag2', source=self.source, line_width=2, color=self.get_color(6, False), visible=False, legend_label="Returns_Lag2")
        # Plot the Returns_Lag3
        p.line('date', 'Returns_Lag3', source=self.source, line_width=2, color=self.get_color(7, False), visible=False, legend_label="Returns_Lag3")
        # Plot the Returns_Lag4
        p.line('date', 'Returns_Lag4', source=self.source, line_width=2, color=self.get_color(8, False), visible=False, legend_label="Returns_Lag4")
        # Plot the Returns_Lag5
        p.line('date', 'Returns_Lag5', source=self.source, line_width=2, color=self.get_color(9, False), visible=False, legend_label="Returns_Lag5")

        # Plot the ROC
        p.line('date', 'ROC', source=self.source, line_width=2, color=self.get_color(9, False), visible=False, legend_label="ROC")

        # Plot the Volume_Change
        p.line('date', 'Volume_Change', source=self.source, line_width=2, color=self.get_color(1, False), visible=False, legend_label="Volume_Change")
        # Plot the Volume_MA
        p.line('date', 'Volume_MA', source=self.source, line_width=2, color=self.get_color(2, False), visible=False, legend_label="Volume_MA")
        # Plot the Volume_MA
        p.line('date', 'Volume_MA', source=self.source, line_width=2, color=self.get_color(3, False), visible=False, legend_label="Volume_MA")
        # Plot the DayOfWeek
        p.line('date', 'DayOfWeek', source=self.source, line_width=2, color=self.get_color(4, False), visible=False, legend_label="DayOfWeek")

        # Plot LowPass Filter
        p.line('date', 'Lowpass', source=self.source, line_width=2, color=self.get_color(1, False), visible=False, legend_label="Lowpass")
        # Plot Highpass Filter
        p.line('date', 'Highpass', source=self.source, line_width=2, color=self.get_color(2, False), visible=False, legend_label="Highpass")




        # Configure hover tool
        hover = HoverTool(
            tooltips=[
                ('Date', '@date{%F %T}'),
                ('Open', '@open{0.2f}'),
                ('High', '@high{0.2f}'),
                ('Low', '@low{0.2f}'),
                ('Close', '@close{0.2f}'),
                # (short_ema_period_column_name, f'@{short_ema_period_column_name}{{0.000000f}}'),  # Explicit 6 zeros
                # (long_ema_period_column_name, f'@{long_ema_period_column_name}{{0.000000f}}')
            ],
            formatters={
                '@date': 'datetime'
            },
            mode='vline'
        )

        # Add hover tool to plot
        p.add_tools(hover)
        
        
        
        # Add an event listener for tap (click) events on the chart
        p.on_event(Tap, self.on_plot_click)

        # Enable clicking to hide/show lines
        p.legend.click_policy = "hide"
        
        # p.legend.location = "top_left"
        p.legend.ncols = 2  # Organize into 2 columns
        p.legend.location = "right"
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

    def Buy_on_submit(self, event):
        """Handle the Buy Submit button click: update BuySignal and refresh the plot."""
        if self.selected_date is None:
            self.html_pane.object = "<b>Please click on the chart to select a date first!</b>"
            return
        
        Buy_radio_value = self.Buy_radio_button_group.value  # This will be 0 or 1 based on selection
        
        # Print the selected date, text input, and radio button value to the console
        print(f"Buy Submitted - Date: {self.selected_date}, Radio: {Buy_radio_value}")
        
        # Update the BuySignal column by adding the radio value to the selected date's BuySignal
        if self.selected_date in self.df.index:
            self.df.loc[self.selected_date, 'BuySignal'] = Buy_radio_value
        else:
            print("Selected date not in DataFrame!")
        
        # Refresh the buy signal data source so that the plot updates accordingly
        self.buy_signal_source.data = self.Buy_get_signal_data()

        # Refresh the sell signal data source so that the plot updates accordingly
        self.sell_signal_source.data = self.Sell_get_signal_data()
        
        # Optionally update the plot title to reflect changes
        self.plot.title.text = "OHLCV Data (Updated with Signal)"

    def Sell_on_submit(self, event):
        """Handle the Sell Submit button click: update SellSignal and refresh the plot."""
        if self.selected_date is None:
            self.html_pane.object = "<b>Please click on the chart to select a date first!</b>"
            return
        
        Sell_radio_value = self.Sell_radio_button_group.value  # This will be 0 or 1 based on selection
        
        # Print the selected date, text input, and radio button value to the console
        print(f"Sell Submitted - Date: {self.selected_date}, Radio: {Sell_radio_value}")
        
        # Update the SellSignal column by adding the radio value to the selected date's SellSignal
        if self.selected_date in self.df.index:
            self.df.loc[self.selected_date, 'SellSignal'] = Sell_radio_value
        else:
            print("Selected date not in DataFrame!")
        
        # Refresh the signal data source so that the plot updates accordingly
        self.sell_signal_source.data = self.Sell_get_signal_data()
        
        # Optionally update the plot title to reflect changes
        self.plot.title.text = "OHLCV Data (Updated with Signal)"
        

    def Buy_on_save(self, event):
        """Handle the Save button click: Output final df to console for validation."""
        filtered_df = self.df[self.df['BuySignal'].isin([0, 1])]
        print(filtered_df)
        print(self.df)

        # TechnicalIndicators.add_lagged_indicators(df, ['BuySignal', 'SellSignal'], lags=[1, 2, 3, 4, 5])

        # Save dataframe to file
        self.dataManager.save_dataframe_as_pickle(self.df)

    def Sell_on_save(self, event):
        """Handle the Save button click: Output final df to console for validation."""
        filtered_df = self.df[self.df['SellSignal'].isin([0, 1])]
        print(filtered_df)
        print(self.df)

        # TechnicalIndicators.add_lagged_indicators(df, ['BuySignal', 'SellSignal'], lags=[1, 2, 3, 4, 5])

        # Save dataframe to file
        self.dataManager.save_dataframe_as_pickle(self.df)



    def show(self):
        """Return the complete Panel layout for serving."""
        return self.layout


# Main execution block
if __name__ == '__main__':


    dataManager = DataManager()
    technicalIndicators = TechnicalIndicators()
    # add EMA indicators values to datadrame
    # dataManager.long_ema_period = 100
    # dataManager.short_ema_period = 13

    root_dir = '/Users/chrisjackson/Desktop/DEV/python/data/1m/TSLA'
    dataManager.pickleFilePath = '/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl'

    # set EMA Short and Long Periods for the EMA indicators
    # short_ema_period, long_ema_period = [13, 100]

    df = dataManager.load_dataframe_from_pickle(dataManager.pickleFilePath)

    if df is not None:
        print(f"Loading data from File '{dataManager.pickleFilePath}'")

        print(df.info(), df.columns)
        # df.glimpse()

    else:
        print(f"File '{dataManager.pickleFilePath}' does not exist. Loading from CSV files")
        df = dataManager.build_df_from_directory(root_dir, 500)

        df = df.drop(columns=['VolumeWeighted'])

        # technicalIndicators.add_ema(df, dataManager.short_ema_period, "short")
        # technicalIndicators.add_ema(df, dataManager.long_ema_period, "long")

        technicalIndicators.add_all_features(df)

        print(df.info(), df.columns)

    app = StockApp(df, dataManager)

    # Serve the app (use 'panel serve <script_name>.py' to run this script)
    pn.serve(app.show, show=True)
