import logging
import os

# Import the classes from their respective files
from backtest import Backtest, DataCollector, CrossOverStrategy
from StockVisualizer import StockVisualizer

def clear_console():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For Mac and Linux
    else:
        _ = os.system('clear')

# Main execution block
if __name__ == '__main__':

    # clear_console()

    # Configure logging
    # logging.basicConfig(level=logging.INFO)
    
    # Initialize collector for Apple
    collector = DataCollector('TSLA')

    try:
        # Get 5-minute data for last 5 days
        symbol, data = collector.fetch_data_history(interval='1m', period=None, start='2025-02-24', end='2025-02-27')

        # fetch_data_download
        # data = collector.fetch_data_download(interval='5m', period=None, start='2025-02-19', end='2025-02-23')

    except Exception as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"\nCaught expected error: {e}")

    bt = Backtest(symbol, data, CrossOverStrategy, cash=10000, commission=0.001,
                strategy_params={'short_period': 10, 'long_period': 50})
    resultsList = bt.run()

    visualizer = StockVisualizer(bt, resultsList, trades=bt.get_trades_as_dicts_timestamped_types())
    visualizer.show()
