import pandas as pd
import pandas_ta as ta
from pandas_ta import macd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Optional, Union
from zoneinfo import ZoneInfo
import sys

# Trade class remains unchanged
class Trade:
    """Represents a closed trade with its attributes."""
    def __init__(self, direction, entry_time, exit_time, entry_price, exit_price, size, profit):
        self.direction = direction
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.size = size
        self.profit = profit

    def __repr__(self):
        return (
            f"\n\nTrade(\n"
            f"  direction={self.direction},\n"
            f"  entry_time={self.entry_time},\n"
            f"  exit_time={self.exit_time},\n"
            f"  entry_price={self.entry_price},\n"
            f"  exit_price={self.exit_price},\n"
            f"  size={self.size},\n"
            f"  profit={self.profit}\n)"
        )

class Position:
    """
    Manages open positions and executes orders, tracking positions using dictionaries.

    Attributes:
        long_positions (dict): {position_id: {'size': size, 'price': price, 'time': time}}
        short_positions (dict): {position_id: {'size': size, 'price': price, 'time': time}}
        position_id_counter (int): Counter for generating unique position IDs
    """
    def __init__(self):
        self.long_positions = {}  # Dictionary instead of list
        self.short_positions = {}  # Dictionary instead of list
        self.position_id_counter = 0  # For unique position IDs

    def execute_order(self, size, price, time, commission):
        """
        Execute a market order, closing existing positions and opening new ones as needed.

        Args:
            size (float): Order size (positive for buy, negative for sell)
            price (float): Execution price
            time (pd.Timestamp): Execution timestamp
            commission (float): Commission rate

        Returns:
            list: List of Trade objects (closed trades)
        """

        closed_trades = []  # list
        if size > 0:  # Buy order
            # Close short positions first
            while size > 0 and self.short_positions:
                position_id, position = next(iter(self.short_positions.items()))  # Get first position (FIFO)
                close_size = min(size, position['size'])
                entry_price = position['price']
                exit_price = price
                trade_profit = close_size * (entry_price - exit_price) - commission * close_size * (entry_price + exit_price)
                trade = Trade('short', position['time'], time, entry_price, exit_price, close_size, trade_profit)
                closed_trades.append(trade)
                position['size'] -= close_size
                if position['size'] == 0:
                    del self.short_positions[position_id]  # Remove fully closed position
                size -= close_size
            if size > 0:
                position_id = self.position_id_counter
                self.position_id_counter += 1
                self.long_positions[position_id] = {'size': size, 'price': price, 'time': time}  # New long position
        elif size < 0:  # Sell order
            size = -size
            # Close long positions first
            while size > 0 and self.long_positions:
                position_id, position = next(iter(self.long_positions.items()))  # Get first position (FIFO)
                close_size = min(size, position['size'])
                entry_price = position['price']
                exit_price = price
                trade_profit = close_size * (exit_price - entry_price) - commission * close_size * (entry_price + exit_price)
                trade = Trade('long', position['time'], time, entry_price, exit_price, close_size, trade_profit)
                closed_trades.append(trade)
                position['size'] -= close_size
                if position['size'] == 0:
                    del self.long_positions[position_id]  # Remove fully closed position
                size -= close_size
            if size > 0:
                position_id = self.position_id_counter
                self.position_id_counter += 1
                self.short_positions[position_id] = {'size': size, 'price': price, 'time': time}  # New short position

        return closed_trades
    

class Strategy:
    """Base class for defining trading strategies."""
    def __init__(self, backtest):
        self.backtest = backtest
        self.i = 0

    def init(self):
        """Initialize the strategy (e.g., precompute indicators)."""
        pass

    def next(self, bar):
        """place buy/sell orders."""
        pass

    @property
    def data(self):
        return self.backtest.data.iloc[:self.i + 1]

    @property
    def current_position(self):
        return self.backtest.position_series.iloc[self.i]

    def buy(self, size):
        """Place a buy order using Backtest's place_order method."""
        self.backtest.pending_orders[self.backtest.order_id_counter] = {'type': 'buy', 'size': size}
        self.backtest.order_id_counter += 1

    def sell(self, size):
        """Place a sell order using Backtest's place_order method."""
        self.backtest.pending_orders[self.backtest.order_id_counter] = {'type': 'sell', 'size': -size}
        self.backtest.order_id_counter += 1

class Backtest:
    """
    Main class for running a backtest, using dictionaries for trades and orders.

    Attributes:
        trades (dict): {trade_id: Trade object}
        pending_orders (dict): {order_id: size}
        trade_id_counter (int): For unique trade IDs
        order_id_counter (int): For unique order IDs
    """
    def __init__(self, symbol, data, strategy, cash=10000, commission=0.0, strategy_params={}):
        self.symbol = symbol
        self.data = data
        self.indicator_values = pd.DataFrame()
        self.strategy_class = strategy
        self.initial_cash = cash
        self.cash = cash
        self.commission = commission
        self.strategy_params = strategy_params
        self.position = Position()
        self.equity_curve = pd.Series(index=data.index, dtype=float)
        self.position_series = pd.Series(index=data.index, dtype=float)
        self.trades = {}  # Dictionary instead of list
        self.trade_id_counter = 0  # For unique trade IDs
        self.order_id_counter = 0  # For unique order IDs
        self.pending_orders = {}  # Dictionary instead of list

    # def place_order(self, size):
    #     """Place an order with a unique ID."""
    #     order_id = self.order_id_counter
    #     self.order_id_counter += 1
    #     self.pending_orders[order_id] = size

    #     print(f"\n\n\n----------------self.pending_orders--------------------------")
    #     print(self.pending_orders)
    #     print(f"------------------------------------------------------\n\n")

    def run(self):
        """
        Run the backtest simulation and return performance statistics.
        """
        self.strategy = self.strategy_class(self, **self.strategy_params)
        self.strategy.init() #run the numbers on indicators and store their values in self.indicator_values

        for t in range(len(self.data)):
            price = self.data['Open'].iloc[t]
            time = self.data.index[t]

            # Execute pending orders
            for order_id, order in self.pending_orders.items():
                if order['size'] > 0:  # Buy
                    total_cost = order['size'] * price * (1 + self.commission)
                    self.cash -= total_cost
                elif order['size'] < 0:  # Sell
                    total_proceeds = (-order['size']) * price * (1 - self.commission)
                    self.cash += total_proceeds

                closed_trades = self.position.execute_order(order['size'], price, time, self.commission)

                for trade in closed_trades:
                    trade_id = self.trade_id_counter
                    self.trade_id_counter += 1
                    self.trades[trade_id] = trade  # Add to dictionary

            self.pending_orders = {}  # Reset to empty dict

            # Update position and equity
            long_position = sum(position['size'] for position in self.position.long_positions.values())
            short_position = sum(position['size'] for position in self.position.short_positions.values())
            position_size = long_position - short_position
            self.position_series.iloc[t] = position_size
            self.equity_curve.iloc[t] = self.cash + position_size * self.data['Close'].iloc[t]

            self.strategy.i = t
            self.strategy.next(self.data.iloc[t]) # strategy logic for buy sell on each bar


        # Compute performance statistics
        start = self.data.index[0]
        end = self.data.index[-1]
        duration = (end - start).total_seconds() / 60
        equity_final = self.equity_curve.iloc[-1]
        equity_peak = self.equity_curve.max()
        total_return = (equity_final / self.initial_cash) - 1

        daily_returns = self.equity_curve.pct_change().dropna()
        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve / running_max - 1)
        max_drawdown = drawdown.min()

        exposure_time = (self.position_series != 0).mean()
        number_of_trades = len(self.trades)
        if number_of_trades > 0:
            win_rate = sum(trade.profit > 0 for trade in self.trades.values()) / number_of_trades
            best_trade = max(trade.profit for trade in self.trades.values())
            worst_trade = min(trade.profit for trade in self.trades.values())
            avg_trade = sum(trade.profit for trade in self.trades.values()) / number_of_trades
            trade_durations = [(trade.exit_time - trade.entry_time).total_seconds() / 60 for trade in self.trades.values()]
            max_trade_duration = max(trade_durations)
            avg_trade_duration = sum(trade_durations) / number_of_trades
            positive_profits = sum(trade.profit for trade in self.trades.values() if trade.profit > 0)
            negative_profits = -sum(trade.profit for trade in self.trades.values() if trade.profit < 0)
            profit_factor = positive_profits / negative_profits if negative_profits > 0 else np.inf
        else:
            win_rate = np.nan
            best_trade = np.nan
            worst_trade = np.nan
            avg_trade = np.nan
            max_trade_duration = np.nan
            avg_trade_duration = np.nan
            profit_factor = np.inf

        stats = {
            'Start': start,
            'End': end,
            'Duration (minutes)': duration,
            'Exposure Time': exposure_time,
            'Equity Final': equity_final,
            'Equity Peak': equity_peak,
            'Return': total_return,
            'Max. Drawdown': max_drawdown,
            '# Trades': number_of_trades,
            'Win Rate': win_rate,
            'Best Trade': best_trade,
            'Worst Trade': worst_trade,
            'Avg. Trade': avg_trade,
            'Max. Trade Duration (minutes)': max_trade_duration,
            'Avg. Trade Duration (minutes)': avg_trade_duration,
            'Profit Factor': profit_factor,
        }
        return pd.Series(stats)
    

    
    def get_trades_as_dicts_timestamped_types(self):

        trade_data = [
            {
                'entry_time': pd.to_datetime(trade.entry_time, utc=False),
                'exit_time': pd.to_datetime(trade.exit_time, utc=False),
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': trade.direction
            }
            for trade in self.trades.values()
        ]

        return trade_data

    
class DataCollector:
    """
    A class to collect intraday financial data from Yahoo Finance using yfinance.
    
    Handles data fetching, validation, and basic preprocessing for intraday trading applications.
    
    Parameters:
    symbol (str): The ticker symbol to fetch data for (e.g., 'AAPL', 'BTC-USD')
    logger (logging.Logger, optional): Logger instance for logging (default: console logging)
    
    Attributes:
    valid_intervals (list): Allowed time intervals ('1m', '5m', '15m', '1h')
    valid_periods (list): Allowed period values ('1d', '5d')
    
    Methods:
    fetch_data: Main method to retrieve market data
    """
    
    valid_intervals = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
    valid_periods = [None, '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']

    def __init__(self, symbol: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataCollector with a security symbol.
        
        Args:
            symbol: Uppercase ticker symbol recognized by Yahoo Finance
            logger: Preconfigured logger instance (optional)
        """
        self.symbol = symbol.upper()
        self.logger = logger or logging.getLogger(__name__)
        # logging.disable(50)
        
        # Configure basic logging if no logger provided
        if not logger:
            logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def fetch_data_history(self,
                 interval: str = '5m',
                 period: Optional[str] = None,
                 start: Optional[Union[str, datetime]] = None,
                 end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch intraday price data from Yahoo Finance.
        
        Args:
            interval: Time interval between prices - must be one of:
                      '1m', '5m', '15m', '1h' (default: '5m')
            period: Time period to fetch, overrides start/end:
                    '1d' = 1 day, '5d' = 5 days (max allowed by Yahoo Finance)
            start: Start datetime (string or datetime object)
            end: End datetime (string or datetime object)
        
        Returns:
            pd.DataFrame: DataFrame with OHLCV data and DatetimeIndex
            
        Raises:
            ValueError: For invalid parameters or data validation failures
            RuntimeError: If data fetch fails from Yahoo Finance
        """

        """
        yfinance.Ticker.history documentation
        :Parameters:
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
        start: str
            Download start date string (YYYY-MM-DD) or _datetime.
            Default is 1900-01-01
        end: str
            Download end date string (YYYY-MM-DD) or _datetime.
            Default is now
        prepost : bool
            Include Pre and Post market data in results?
            Default is False
        auto_adjust: bool
            Adjust all OHLC automatically? Default is True
        back_adjust: bool
            Back-adjusted data to mimic true historical prices
        proxy: str
            Optional. Proxy server URL scheme. Default is None
        rounding: bool
            Round values to 2 decimal places?
            Optional. Default is False = precision suggested by Yahoo!
        tz: str
            Optional timezone locale for dates.
            (default data is returned as non-localized dates)
        timeout: None or float
            If not None stops waiting for a response after given number of
            seconds. (Can also be a fraction of a second e.g. 0.01)
            Default is None.
        **kwargs: dict
            debug: bool
                Optional. If passed as False, will suppress
                error message printing to console.
        """

        self.interval = interval
        self.period = period

        # Validate input parameters
        if self.interval not in self.valid_intervals:
            raise ValueError(f"Invalid interval. Use one of {self.valid_intervals}")
            
        if self.period and self.period not in self.valid_periods:
            raise ValueError(f"Invalid period. Use one of {self.valid_periods} or None")
            
        # Yahoo Finance API constraints
        if interval == '1m' and (self.period not in ['1d', None] or 
                                (self.period == '5d' and self.interval in ['1m', '5m'])):
            raise ValueError(f"1m data only available for maximum 7 days, 5m for 60 days")
        
        # Convert datetime objects to strings
        if isinstance(start, datetime):
            start = start.strftime('%Y-%m-%d')
        if isinstance(end, datetime):
            end = end.strftime('%Y-%m-%d')
        
        self.logger.info(f"Fetching {self.symbol} data - Interval: {self.interval}, "
                       f"Period: {self.period}, Start: {start}, End: {end}")
        
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Fetch data using yfinance
            df = ticker.history(
                # period=self.period,
                interval=self.interval,
                start=start,
                end=end,
                prepost=False,    # Exclude pre/post market data
                auto_adjust=False # Keep original OHLC values
            )

            df.index = pd.to_datetime(df.index)
            # df.index = df.index.tz_convert("US/Eastern")
            df.index = df.index.tz_convert(ZoneInfo("America/New_York"))
            # df.index.tz_localize(None)
            # df.index = df.index.tz_convert("UTC")  # Convert Eastern Time to UTC
            
            # Validate returned data
            if df.empty:
                raise ValueError(f"No data returned for {self.symbol} with given parameters")
            
            # add the self.interval column to df
            df['interval'] = self.interval
                
            # Basic data cleaning
            # df = self._process_data(df)
            
            self.logger.info(f"Successfully fetched {len(df)} bars")
            return self.symbol, df
            
        except Exception as e:
            error_msg = f"Failed to fetch data: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        

    def fetch_data_download(self):
        """
        yf_download(self, period="1mo", interval="1d",
            start=None, end=None, prepost=False, actions=True,
            auto_adjust=True, back_adjust=False,
            proxy=None, rounding=False, tz=None, timeout=None, **kwargs):

        yfinance.download documentation
        :Parameters:
            period : str
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                Either Use period parameter or use start and end
            interval : str
                Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                Intraday data cannot extend last 60 days
            start: str
                Download start date string (YYYY-MM-DD) or _datetime.
                Default is 1900-01-01
            end: str
                Download end date string (YYYY-MM-DD) or _datetime.
                Default is now
            prepost : bool
                Include Pre and Post market data in results?
                Default is False
            auto_adjust: bool
                Adjust all OHLC automatically? Default is True
            back_adjust: bool
                Back-adjusted data to mimic true historical prices
            proxy: str
                Optional. Proxy server URL scheme. Default is None
            rounding: bool
                Round values to 2 decimal places?
                Optional. Default is False = precision suggested by Yahoo!
            tz: str
                Optional timezone locale for dates.
                (default data is returned as non-localized dates)
            timeout: None or float
                If not None stops waiting for a response after given number of
                seconds. (Can also be a fraction of a second e.g. 0.01)
                Default is None.
            **kwargs: dict
                debug: bool
                    Optional. If passed as False, will suppress
                    error message printing to console.
        """
        return False
        

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the raw data from Yahoo Finance.
        
        1. Remove missing values
        2. Ensure proper datetime index
        3. Validate required columns
        
        Args:
            df: Raw DataFrame from yfinance
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
            
        # Convert index to UTC and remove timezone info
        df.index = df.index.tz_localize(None)
        
        # Handle missing data
        df = df.dropna()
        df = df[~df.index.duplicated(keep='first')]
        
        # Forward-fill missing values (intraday specific)
        df = df.asfreq('min').ffill()  # Changed 'T' to 'min'
        
        return df
    
class Managment(Strategy):
    
    def __init__(self, backtest, settings):
        
        super().__init__(backtest)
        self.settings = settings


    def init(self):
        pass

    def set_target_profit(self, ticker, entry_price, target_profit_percentage):
        """
        Sets a target profit order for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol (e.g., "AAPL").
            entry_price (float): The price at which the stock was purchased.
            target_profit_percentage (float): The percentage above the entry price
            at which to trigger the target profit.
        """
        pass

    def set_stop_loss(ticker, entry_price, stop_loss_percentage):
        """
        Sets a stop-loss order for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol (e.g., "AAPL").
            entry_price (float): The price at which the stock was purchased.
            stop_loss_percentage (float): The percentage below the entry price 
                                        at which to trigger the stop-loss.

        Returns:
            float: The stop-loss price.
        """
        stop_loss_price = entry_price * (1 - stop_loss_percentage)
        return stop_loss_price

class CrossOverStrategy(Strategy):
    """
    Intraday trading strategy based on EMA crossovers.

    Buys when the short EMA crosses above the long EMA and sells when it crosses below.
    Designed for intraday data with multiple trades possible within a day.

    Attributes:
        short_period (int): Number of bars for the short EMA.
        long_period (int): Number of bars for the long EMA.
        ema_short (pd.Series): Short-term EMA of closing prices.
        ema_long (pd.Series): Long-term EMA of closing prices.
    """
    def __init__(self, backtest, short_period=10, long_period=200):
        """
        Initialize the strategy with EMA periods.

        Args:
            backtest (Backtest): The backtest instance.
            short_period (int): Period for the short EMA, default 10.
            long_period (int): Period for the long EMA, default 200.
        """
        super().__init__(backtest)
        self.short_period = short_period
        self.long_period = long_period
        self.i = 0


    def init(self):
        """Compute short and long EMAs and add them to self.indicator_values."""

        # self.long_period is the indicator needs the largest window of data
        if  self.backtest.data["Close"].notna().sum() < self.long_period:
            print(f"\n\nDownloaded data points: {self.backtest.data['Close'].notna().sum()}")
            print(f"Not enough data to calculate EMA_{self.long_period} values.\n\n")
            sys.exit(0)  # Successful exit
        else:

            self.backtest.indicator_values = pd.DataFrame(index=self.backtest.data.index)

            self.backtest.indicator_values[f'EMA_{self.short_period}'] = (
                ta.ema(self.backtest.data['Close'], length=self.short_period)
                .reindex(self.backtest.data.index)  # Ensure same index
            )

            self.backtest.indicator_values[f'EMA_{self.long_period}'] = (
                ta.ema(self.backtest.data['Close'], length=self.long_period)
                .reindex(self.backtest.data.index)  # Ensure same index
            )

            #MACD values
            macd_df = macd(self.backtest.data['Close'], fast=12, slow=26, signal=9)
            self.backtest.indicator_values['MACD'] = macd_df['MACD_12_26_9']
            self.backtest.indicator_values['MACDs'] = macd_df['MACDs_12_26_9']
            self.backtest.indicator_values['MACDh'] = macd_df['MACDh_12_26_9']

            

            self.backtest.indicator_values['Close'] = self.backtest.data['Close']

            # Drop rows where EMA_long_period is NaN
            # self.backtest.indicator_values = self.backtest.indicator_values.dropna(subset=[f'EMA_{self.long_period}'])



    def next(self, bar):
        """
        Check for EMA crossover conditions and place buy/sell orders.

        Args:
            bar (pd.Series): Current bar data.
        """

        bar_datetime = pd.to_datetime(bar.name)  # Ensure it's in datetime format
        df = self.backtest.indicator_values  # For readability

        # Ensure bar_datetime has the same timezone as df's index, or remove timezone
        bar_datetime = bar_datetime.tz_localize('US/Eastern', ambiguous='NaT') if bar_datetime.tzinfo is None else bar_datetime
        # Alternatively, strip the timezone from df's index if you don't care about timezones:
        # df.index = df.index.tz_localize(None)

        # Ensure the dataframe has the EMA columns
        short_col = f'EMA_{self.short_period}'
        long_col = f'EMA_{self.long_period}'
        
        # Drop rows where either EMA is NaN (vectorized NaN handling)
        df = df.dropna(subset=[short_col, long_col]).copy()
        
        # Create shifted columns for previous EMA values (vectorized equivalent of prev_ema_short/long)
        df['prev_short'] = df[short_col].shift(1)
        df['prev_long'] = df[long_col].shift(1)
        
        # Detect crossovers (vectorized conditions)
        df['cross_above'] = (df['prev_short'] <= df['prev_long']) & (df[short_col] > df[long_col])
        df['cross_below'] = (df['prev_short'] >= df['prev_long']) & (df[short_col] < df[long_col])
        
        # Check if the current bar has a crossover
        if bar_datetime in df.index:
            row = df.loc[bar_datetime]
            if row['cross_above']:
                self.buy(1)  # Buy on crossover above
            elif row['cross_below']:
                self.sell(1)  # Sell on crossover below
