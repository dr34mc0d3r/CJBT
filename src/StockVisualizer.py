"""Panel page for backtest app"""

import os
import pandas as pd
from pandas_ta import macd
# import hvplot.pandas
import hvplot.pandas as hv
import panel as pn
from bokeh import events
from bokeh.plotting import figure
# from bokeh.models import Span, ColumnDataSource
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, LabelSet, WheelZoomTool, Span, CustomJS, Range1d
from bokeh.events import MouseMove
from bokeh.models.formatters import CustomJSTickFormatter
import numpy as np
from zoneinfo import ZoneInfo

pn.extension()

class StockVisualizer:
    def __init__(self, backtest, resultsList=None, trades=None):
        self.backtest = backtest
        self.df = backtest.data
        self.indicator_values = backtest.indicator_values
        self.trades = trades or []  # Initialize with empty list if no trades provided
        self.resultsList = resultsList

        self.strategy_params = backtest.strategy_params
        self._validate()

        self.df_all_trades = pd.DataFrame()

    def _validate(self):
        # Existing validation code...
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column.")
        
        # Validate trades if provided
        if self.trades:
            required_trade_keys = ['entry_time', 'exit_time', 'entry_price', 'exit_price']
            for trade in self.trades:
                if not isinstance(trade, dict):
                    raise ValueError("Each trade must be a dictionary.")
                for key in required_trade_keys:
                    if key not in trade:
                        raise ValueError(f"Trade missing required key: {key}")
                if not isinstance(trade['entry_time'], pd.Timestamp) or not isinstance(trade['exit_time'], pd.Timestamp):
                    raise ValueError("Trade 'entry_time' and 'exit_time' must be pd.Timestamp.")

    def _create_trade_markers_one_df_out(self):

        if self.trades:
            all_trades = []
            for trade in self.trades:
                all_trades.append({
                    'time': trade['entry_time'],
                    'trade_price': trade['entry_price'],
                    'trade_type': 'Buy'
                    }
                )

                all_trades.append({
                    'time': trade['exit_time'],
                    'trade_price': trade['exit_price'],
                    'trade_type': 'Sell'
                    }
                )

            self.df_all_trades = pd.DataFrame(all_trades).sort_values(by='time').reset_index(drop=True)

            # First, initialize the Total column with 0.000000
            self.df_all_trades['last_trade_profit'] = 0.000000

            # Create a variable to store the last buy price
            last_buy_price = 0.0

            # Iterate through the dataframe
            for index, row in self.df_all_trades.iterrows():
                if row['trade_type'] == 'Buy':
                    # Store the buy price for the next sell calculation
                    last_buy_price = row['trade_price']
                elif row['trade_type'] == 'Sell':
                    # Calculate profit/loss (Sell price - previous Buy price)
                    # Add to running total from previous row if it exists
                    if index > 0:
                        previous_total = self.df_all_trades.loc[index-1, 'last_trade_profit']
                        self.df_all_trades.loc[index, 'last_trade_profit'] = previous_total + (row['trade_price'] - last_buy_price)

            # If you want to round the results to a specific number of decimals
            self.df_all_trades['last_trade_profit'] = self.df_all_trades['last_trade_profit'].round(6)


    def create_indicator_count(self, max_count, dataframe):
        # Take the last max_count rows from the input dataframe
        indicator_count = dataframe.tail(max_count).copy()
        
        # Convert datetime to milliseconds since epoch (numeric)
        indicator_count.index = pd.to_datetime(indicator_count.index)

        # Create the new dataframe with required columns
        indicator_count = indicator_count[['Close']].rename(columns={'Close': 'y_axis'})

        indicator_count['x_timestamp'] = self.df.index #indicator_count.index
        
        # Create tick_label column with descending integers from max_count-1 to 0
        indicator_count['tick_label'] = range(max_count, 0, -1)

        return indicator_count
    

    def create_showind_div(self):
        # used in function chart_tab_content
        min_date = self.df.index.min()
        max_date = self.df.index.max()
        div_content = pn.pane.HTML(f"""
            <div style='padding: 1px;'>
                <h3>Showing {self.backtest.symbol}</h3>
                <p>Start: {min_date} End: {max_date}</p>
            </div>
        """, width=600)

        return div_content
    
    def create_bokeh_OHLC(self):
        color_map = {
            'green': (184, 184, 184),  # RGB for green
            'red': (92, 92, 92),    # RGB for red
            'grid_line_color': (112, 112, 112)
        }

        # Extract min/max dates from dataDF
        min_date = self.df.index.min()
        max_date = self.df.index.max()
        # Define figure with range limiting
        x_range = Range1d(start=min_date, end=max_date)

        # get the interval
        interval = self.df['interval'].iloc[0]
        interval_millisec = 0
        if interval == "1m":
            interval_millisec = 50000
        elif interval == "5m":
            interval_millisec = 50000 * 5
        elif interval == "15m":
            interval_millisec = 50000 * 15

        # create a new plot with a title and axis labels
        p = figure(x_axis_type="datetime",
                   x_range=x_range,
                   y_axis_label="Price",
                   sizing_mode="stretch_width",
                   background_fill_color=(41, 41, 41),
                )
        
        # Apply min_interval to prevent excessive zooming
        p.x_range.min_interval = interval_millisec
        
        p.xaxis.formatter = DatetimeTickFormatter(
            days="%Y-%m-%d",
            hours="%Y-%m-%d %H:%M",
            minutes="%H:%M",
            seconds="%H:%M:%S"
        )

        p.xaxis.major_label_orientation = 0.5  # Helps with label visibility

        p.xaxis.formatter = CustomJSTickFormatter(code="""
        var date = new Date(tick);
        var year = date.getFullYear();
        var month = date.getMonth() + 1;
        var day = date.getDate();
        var hours = date.getHours();
        var minutes = date.getMinutes();
        var seconds = date.getSeconds();
        return year + "-" + (month < 10 ? "0" + month : month) + "-" + 
            (day < 10 ? "0" + day : day) + " " + 
            (hours < 10 ? "0" + hours : hours) + ":" + 
            (minutes < 10 ? "0" + minutes : minutes) + ":" + 
            (seconds < 10 ? "0" + seconds : seconds);
        """)
        
        # p.grid.visible = False
        # Customize the grid lines for the x-axis
        p.xgrid.grid_line_color = color_map['grid_line_color']  # Set the color to blue
        p.xgrid.grid_line_dash = "dotted"  # Set the style to dotted

        # Customize the grid lines for the y-axis
        p.ygrid.grid_line_color = color_map['grid_line_color']   # Set the color to red
        p.ygrid.grid_line_dash = "dotted"  # Set the style to dotted

        # Optional: Adjust grid line width if desired
        p.xgrid.grid_line_width = 1.0
        p.ygrid.grid_line_width = 1.0

        self.df['Color'] = np.where(self.df['Close'] > self.df['Open'], 'green', 'red')
        # Map 'green' and 'red' to RGB values
        
        # Apply the mapping
        self.df['Color_RGB'] = self.df['Color'].map(color_map)

        # body
        p.rect(x=self.df.index, 
               y=(self.df['Open'] + self.df['Close'])/2, 
               width=interval_millisec, #50000 for 1min data
               height=abs(self.df['Open'] - self.df['Close']), 
               fill_color=self.df['Color_RGB'],
               line_color=self.df['Color_RGB']
               )
        
        # wicks
        p.rect(x=self.df.index, 
               y=(self.df['High'] + self.df['Low'])/2, 
               width=1000, #1 second width
               height=abs(self.df['High'] - self.df['Low']), 
               fill_color=self.df['Color_RGB'],
               line_color=self.df['Color_RGB']
               )

        # add trades to self.df_all_trades 
        self._create_trade_markers_one_df_out()

        p.line(x=self.indicator_values.index, 
            y = self.indicator_values[f"EMA_{self.strategy_params['short_period']}"], 
            legend_label=f"EMA_{self.strategy_params['short_period']}", 
            line_width=2,
            line_color="blue"
        )
        
        p.line(x=self.indicator_values.index, 
            y = self.indicator_values[f"EMA_{self.strategy_params['long_period']}"], 
            legend_label=f"EMA_{self.strategy_params['long_period']}", 
            line_width=2,
            line_color="yellow"
        )

        p.line(x=self.df.index, 
            y = self.df['High'], 
            legend_label="High", 
            line_width=2,
            line_color="red"
        )
        
         # Add trade markers if trades exist
        if self.trades:
            self.df_all_trades = self.df_all_trades.set_index('time')
            self.df = pd.concat([self.df, self.df_all_trades], axis=1)

            GREEN_RGB = (7, 245, 82)
            RED_RGB = (245, 7, 67)
            DEFAULT_RGB = (0, 0, 0)

            self.df['Color'] = self.df['trade_type'].apply(
                lambda x: GREEN_RGB if x == 'Buy' else RED_RGB if x == 'Sell' else DEFAULT_RGB
            )

            p.scatter(
                x=self.df.index,           # DateTimeIndex for x-axis
                y=self.df['trade_price'],        # Price values for y-axis
                size=40,              # Size of the markers
                fill_color=self.df['Color'],   # Color of the markers
                fill_alpha=1,      # Transparency of fill
                line_color='black',   # Outline color
                line_alpha=1,      # Transparency of outline
                line_width=5,
                legend_label='Trade',
                marker="diamond"
            )

        # indicator_countDF = self.create_indicator_count(self.strategy_params['long_period'], self.df)
        indicator_countDF = self.create_indicator_count(len(self.indicator_values), self.df)
        # Create a ColumnDataSource from the DataFrame
        source = ColumnDataSource(indicator_countDF)

        # # Add labels
        label_set = LabelSet(
            x='x_timestamp',
            y=indicator_countDF['y_axis'].max(),
            text='tick_label',
            text_font_size="8pt",
            text_color=color_map['grid_line_color'],
            source=source
            )
        p.add_layout(label_set)

        # Calculate MACD using pandas_ta
        # macd_df = macd(self.df['Close'], fast=12, slow=26, signal=9)
        # self.indicator_values['MACD'] = macd_df['MACD_12_26_9']
        # self.indicator_values['MACDs'] = macd_df['MACDs_12_26_9']
        # self.indicator_values['MACDh'] = macd_df['MACDh_12_26_9']
        
        # Create Bokeh figure for MACD
        p2 = figure(x_axis_type="datetime",
                   x_range=p.x_range,
                   sizing_mode="stretch_width",
                   background_fill_color=(41, 41, 41),
                )
        p2.xaxis.formatter = DatetimeTickFormatter(
            days="%Y-%m-%d",
            hours="%Y-%m-%d %H:%M",
            minutes="%H:%M",
            seconds="%H:%M:%S"
        )

        p2.xaxis.major_label_orientation = 0.5  # Helps with label visibility

        p2.xaxis.formatter = CustomJSTickFormatter(code="""
        var date = new Date(tick);
        var year = date.getFullYear();
        var month = date.getMonth() + 1;
        var day = date.getDate();
        var hours = date.getHours();
        var minutes = date.getMinutes();
        var seconds = date.getSeconds();
        return year + "-" + (month < 10 ? "0" + month : month) + "-" + 
            (day < 10 ? "0" + day : day) + " " + 
            (hours < 10 ? "0" + hours : hours) + ":" + 
            (minutes < 10 ? "0" + minutes : minutes) + ":" + 
            (seconds < 10 ? "0" + seconds : seconds);
        """)

        # p2.grid.visible = False
        # Customize the grid lines for the x-axis
        p2.xgrid.grid_line_color = color_map['grid_line_color']  # Set the color to blue
        p2.xgrid.grid_line_dash = "dotted"  # Set the style to dotted

        # Customize the grid lines for the y-axis
        p2.ygrid.grid_line_color = color_map['grid_line_color']   # Set the color to red
        p2.ygrid.grid_line_dash = "dotted"  # Set the style to dotted

        # Optional: Adjust grid line width if desired
        p2.xgrid.grid_line_width = 1.0
        p2.ygrid.grid_line_width = 1.0
        
        # enable Wheel Zoom Tool
        p.toolbar.active_scroll = p.select_one(WheelZoomTool)
        p2.toolbar.active_scroll = p2.select_one(WheelZoomTool)

        # Plot MACD and Signal lines
        p2.line(self.indicator_values.index, self.indicator_values['MACD'], color='blue', legend_label='MACD')
        p2.line(self.indicator_values.index, self.indicator_values['MACDs'], color='orange', legend_label='Signal')

        # Calculate histogram colors (green for positive, red for negative)
        histogram = self.indicator_values['MACDh']

        # Plot histogram
       
        condition = histogram >= 0
        colors = np.full((len(histogram), 3), color_map['red'], dtype=np.uint8)  # Initialize with red
        colors[condition] = color_map['green']  # Replace with green where condition is True

        p2.vbar(x=self.indicator_values.index,
                top=histogram, 
                width=interval_millisec, 
                color=colors, 
                alpha=0.6)

        # Add zero line
        p2.line(self.indicator_values.index, 0, color='black', line_dash='dashed')

        # Customize the plot
        p2.legend.location = "top_left"
        # p2.xaxis.axis_label = 'Time'
        p2.yaxis.axis_label = 'MACD Value'

        # test marker
        # specific_x = pd.to_datetime('2025-02-19 14:30:00+00:00')  # Example x-coordinate
        # specific_y = 362.00  # Example y-coordinate
        # p.scatter([specific_x], [specific_y], size=10, color="red", legend_label="Specific Point")

        # Add vertical Span glyphs (initially off-screen or at a default position)
        vline1 = Span(location=self.indicator_values.index[0], dimension="height", line_color=color_map['grid_line_color'], line_width=2)
        vline2 = Span(location=self.indicator_values.index[0], dimension="height", line_color=color_map['grid_line_color'], line_width=2)
        # Append the Spans to the renderers list of each plot
        p.renderers.append(vline1)
        p2.renderers.append(vline2)

        hline = Span(location=0, dimension='width', line_color=color_map['grid_line_color'], line_width=2)
        # Append the Spans to the renderers list of each plot
        p.renderers.append(hline)
        p2.renderers.append(hline)

        # JavaScript callback to update the vertical lines based on cursor position
        callback_vertical = CustomJS(args=dict(vline1=vline1, vline2=vline2), code="""
            // Get the x-coordinate of the mouse (in data space)
            const x = cb_obj.x;
            
            // Update the location of the vertical lines on both plots
            vline1.location = x;
            vline2.location = x;
        """)

        # Define the JavaScript callback to update the Span's location
        callback_horizontal = CustomJS(args=dict(hline=hline), code="""
            // cb_obj.y is the y-coordinate of the cursor in data space
            hline.location = cb_obj.y;
        """)

        # Attach the callback to the 'mousemove' event on both plots
        p.js_on_event("mousemove", callback_vertical)
        p2.js_on_event("mousemove", callback_vertical)
        p.js_on_event(MouseMove, callback_horizontal)
        p2.js_on_event(MouseMove, callback_horizontal)

        # # Combine plots vertically
        # pn.Column(pn.pane.Str(self.resultsList), height=400, sizing_mode='stretch_width', scroll=True)
        # layout = column(p1, p2)
        # combined_layout = pn.Column(p, p2, sizing_mode='stretch_width')
        pane1 = pn.pane.Bokeh(p, height=200, sizing_mode='stretch_width')
        pane2 = pn.pane.Bokeh(p2, height=200, sizing_mode='stretch_width')
        combined_layout = pn.Row(pane1, pane2, sizing_mode='stretch_width')
        
        return combined_layout
    
    def indicator_table(self):
        # Temporarily set Pandas display options to show all rows
        pd.set_option('display.max_rows', None)  # No limit on rows
        # Convert DataFrame to string
        df_str = str(self.indicator_values)
        # Reset display option to default (optional, to avoid affecting other outputs)
        pd.reset_option('display.max_rows')

        # Create a div with content using pn.pane.HTML
        # Create a div with content including the variable using an f-string
        div_content = pn.pane.HTML(f"""
            <div style='border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;'>
                <h3>Additional Info</h3>
                <p>Length: {len(self.indicator_values)}</p>
            </div>
        """, width=300)

        # Combine the table and div in a Row layout
        combined_layout = pn.Row(pn.pane.Str(df_str), div_content, sizing_mode='stretch_width')

        # Wrap in a scrollable Column
        scrollable_pane = pn.Column(combined_layout, height=400, sizing_mode='stretch_width', scroll=True)

        return scrollable_pane
    
    def df_table(self):
        # Temporarily set Pandas display options to show all rows
        pd.set_option('display.max_rows', None)  # No limit on rows
        # Convert DataFrame to string
        df_str = str(self.df)
        # Reset display option to default (optional, to avoid affecting other outputs)
        pd.reset_option('display.max_rows')
        # Wrap in a scrollable Column
        scrollable_pane = pn.Column(pn.pane.Str(df_str), height=400, sizing_mode='stretch_width', scroll=True)
        return scrollable_pane
    
    def resultsList_view(self):
        # Wrap in a scrollable Column
        scrollable_pane = pn.Column(pn.pane.Str(self.resultsList), height=400, sizing_mode='stretch_width', scroll=True)
        return scrollable_pane
    
    def trades_view(self):
        # Wrap in a scrollable Column
        scrollable_pane = pn.Column(pn.pane.Str(self.df_all_trades), height=400, sizing_mode='stretch_width', scroll=True)

        # Create a div with content including the variable using an f-string
        div_content = pn.pane.HTML(f"""
            <div style='border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;'>
                <h3>Additional Info</h3>
                <p>Total Profit: ${self.df_all_trades['last_trade_profit'].sum():.2f}</p>
            </div>
        """, width=300)

        # Combine the table and div in a Row layout
        combined_layout = pn.Row(scrollable_pane, div_content, sizing_mode='stretch_width')

        # Wrap in a scrollable Column
        scrollable_pane = pn.Column(combined_layout, height=400, sizing_mode='stretch_width', scroll=True)

        return scrollable_pane
    
    def chart_tab_content(self):
        picker = self.create_showind_div()
        chart = self.create_bokeh_OHLC()
        
        chart_column = pn.Column(
            picker,
            chart,
        )

        return chart_column
    
    def df_tab_content(self):
        mainData = self.df_table()
        indicatorData = self.indicator_table()
        
        chart_column = pn.Column(
            mainData,
            indicatorData,
        )

        return chart_column

    def panel(self):
        p1 = self.chart_tab_content()
        p2 = self.df_tab_content()
        p3 = self.resultsList_view()
        p4 = self.trades_view()

        tabs = pn.Tabs(('Chart', p1), ('DataFrame', p2), ('Results', p3), ('Trades', p4))

        return pn.Column(
            tabs
        ).servable()

    def show(self):

        # Get allowed origins from environment variable
        # allowed_origins = os.getenv('BOKEH_ALLOW_WS_ORIGIN', 'localhost:3000').split(',')
        
        
        pn.serve(self.panel)

        # pn.serve(
        #     {'/': self.panel},
        #     port=3000,
        #     show=False,
        #     title='Simple Line Chart',
        #     address='0.0.0.0',           # Bind to all network interfaces
        #     allow_websocket_origin=["192.168.1.200:3000","173.170.187.120:3000"],
        #     threaded=True,       # Enable threaded server for better concurrency
        #     num_procs=1,         # Number of processes (1 is usually fine for simple apps)
        #     websocket_max_message_size=20*1024*1024  # Increase max message size if needed
        # )