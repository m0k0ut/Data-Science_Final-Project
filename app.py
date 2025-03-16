import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Streamlit app
st.set_page_config(page_title="Identifying Key Support and Resistance In Price Levels", layout="wide")
st.title('Key Support and Resistance In Price Levels')

st.markdown("""
This tool aims to identify key support and resistance price levels in stocks using various algorithmic methods. Each method is detailed below.
""")

with st.expander("Click here to read the description of each method:", expanded=False):
    st.markdown("""
    1. **Pivot Points**: Short-term trend indicators used to determine potential support and resistance levels based on the high, low, and close prices of previous trading sessions.
    2. **Support and Resistance Levels using Rolling Midpoint Range**: Key price points where the stock's price tends to halt its upward or downward trajectory, identified using a rolling window to calculate dynamic support and resistance levels.
    3. **Swing Highs and Lows**: Local maxima and minima used to identify trends and potential reversal points by pinpointing key inflection points on a stock's chart.
    4. **Fibonacci Retracement Levels**: Horizontal lines indicating potential support and resistance levels based on Fibonacci numbers, helping to identify prospective market reversal points.
    5. **Trendlines**: Straight lines drawn to connect two or more price points, helping identify the market trend direction and potential areas of support and resistance.
    6. **Volume Profile**: A charting tool that shows the amount of volume traded at different price levels over a specified period, helping identify areas of high trading activity which can act as support or resistance.
    7. **KMeans Clustering**: A machine learning algorithm used to partition the dataset into clusters, identifying patterns and grouping similar price movements together to highlight significant price levels.
    """)

# Sidebar: How to use and Input Parameters
st.sidebar.title('Input Parameters')

with st.sidebar.expander("How to use:", expanded=False):
    st.markdown("""
    1. **Enter Ticker**: Specify a stock ticker or crypto pair.
    2. **Set Dates**: Choose the date range for analysis.
    3. **Adjust Parameters**: Modify methodology parameters as needed.
    4. **Run Analysis**: Click 'Run' to generate results.
    """)

with st.sidebar.expander("Ticker and Date Settings", expanded=True):
    st.write("Specify the ticker and date range for analysis.")
    ticker = st.text_input('Stock Ticker or Crypto Pair', 'AAPL', help="Enter stock ticker (e.g., AAPL) or crypto pair (e.g., BTC-USD).")
    start_date = st.date_input('Start Date', pd.to_datetime('2023-01-01'))
    end_date = st.date_input('End Date', datetime.now() + timedelta(days=1))

with st.sidebar.expander("Pivot Points and Levels", expanded=True):
    window_period = st.slider('Window Period for Pivot Points and Levels', min_value=10, max_value=60, value=30, help="Set the window period for calculating pivot points and support/resistance levels.")

with st.sidebar.expander("Trendlines and Fibonacci Levels", expanded=True):
    lookback_period = st.slider('Lookback Period for Trendlines and Fibonacci', min_value=10, max_value=60, value=30, help="Set the lookback period for calculating trendlines and Fibonacci retracement levels.")

with st.sidebar.expander("Volume Profile and KMeans", expanded=True):
    n_days = st.slider('Lookback Period for Volume Profile and KMeans (Days)', min_value=30, max_value=365, value=60, help="Set the number of days for calculating volume profile and KMeans clustering.")
    num_clusters = st.slider('Number of Clusters for KMeans', min_value=2, max_value=10, value=3, help="Set the number of clusters for KMeans analysis.")

# Define functions for different analyses
def calculate_pivot_points(df, window):
    # Ensure single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calculate each step explicitly as a Series
    pivot = df['Close'].rolling(window=window).mean()
    low_min = df['Low'].rolling(window=window).min().reindex(df.index, method='ffill')
    high_max = df['High'].rolling(window=window).max().reindex(df.index, method='ffill')
    
    df['Pivot'] = pivot
    df['R1'] = pd.Series(2 * pivot - low_min, index=df.index)
    df['S1'] = pd.Series(2 * pivot - high_max, index=df.index)
    df['R2'] = pd.Series(pivot + (high_max - low_min), index=df.index)
    df['S2'] = pd.Series(pivot - (high_max - low_min), index=df.index)
    
    return df

def find_levels(data, window):
    resistance = data['High'].rolling(window=window).max()
    support = data['Low'].rolling(window=window).min()
    return support, resistance

def check_significant_break(data, support, resistance):
    breaks_above_resistance = (data['Close'] > resistance.shift(1)) & (data['Volume'] > data['Volume'].rolling(window=30).mean())
    breaks_below_support = (data['Close'] < support.shift(1)) & (data['Volume'] > data['Volume'].rolling(window=30).mean())
    return breaks_above_resistance, breaks_below_support

def prepare_data_for_trendlines(data, lookback_period):
    data['Swing_High'] = data['High'][argrelextrema(data['High'].values, np.greater_equal, order=lookback_period)[0]]
    data['Swing_Low'] = data['Low'][argrelextrema(data['Low'].values, np.less_equal, order=lookback_period)[0]]
    return data

def calculate_fibonacci_levels(data, lookback_period):
    high_prices = data["High"].rolling(window=lookback_period).max()
    low_prices = data["Low"].rolling(window=lookback_period).min()
    price_diff = high_prices - low_prices
    levels = np.array([0, 0.236, 0.382, 0.5, 0.618, 0.786, 1])
    fib_levels = low_prices.values.reshape(-1, 1) + price_diff.values.reshape(-1, 1) * levels
    return fib_levels, levels

def calculate_volume_profile(data, n_days):
    filtered_data = data[-n_days:]
    price_bins = np.linspace(filtered_data['Low'].min(), filtered_data['High'].max(), 100)
    volume_profile = [filtered_data['Volume'][(filtered_data['Close'] > price_bins[i]) & (filtered_data['Close'] <= price_bins[i+1])].sum() for i in range(len(price_bins)-1)]
    return price_bins, volume_profile

def calculate_kmeans_clusters(data, n_days, num_clusters):
    filtered_data = data[-n_days:]
    X_time = np.linspace(0, 1, len(filtered_data)).reshape(-1, 1)
    X_price = (filtered_data['Close'].values - np.min(filtered_data['Close'])) / (np.max(filtered_data['Close']) - np.min(filtered_data['Close']))
    X_cluster = np.column_stack((X_time, X_price))
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X_cluster)
    cluster_centers = kmeans.cluster_centers_[:, 1] * (np.max(filtered_data['Close']) - np.min(filtered_data['Close'])) + np.min(filtered_data['Close'])
    return cluster_centers

# Run the analysis
if st.sidebar.button('Run Analysis'):
    # Fetch data with auto_adjust=False and flatten columns if multi-indexed
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if not data.empty:
        # Calculate Pivot Points
        try:
            df_pivot = calculate_pivot_points(data.copy(), window_period)
            df_pivot = df_pivot.dropna()
        except ValueError as e:
            st.error(f"Error in calculate_pivot_points: {e}")
            st.write("DataFrame columns:", data.columns)
            st.write("Sample data:", data.head())
            raise

        # Calculate Support and Resistance Levels
        support, resistance = find_levels(data.copy(), window_period)
        breaks_above_resistance, breaks_below_support = check_significant_break(data.copy(), support, resistance)

        # Calculate Swing Highs and Lows
        data_with_trendlines = prepare_data_for_trendlines(data.copy(), lookback_period)

        # Calculate Fibonacci Retracement Levels
        fib_levels, levels = calculate_fibonacci_levels(data.copy(), lookback_period)

        # Calculate Volume Profile
        price_bins, volume_profile = calculate_volume_profile(data.copy(), n_days)

        # Calculate KMeans Clusters
        cluster_centers = calculate_kmeans_clusters(data.copy(), n_days, num_clusters)

        # Plot Pivot Points
        st.write("### Pivot Points")
        st.markdown("""
        **Pivot Points** are short-term trend indicators used to determine potential support and resistance levels. The central pivot point, as well as derived support and resistance levels, are calculated using the high, low, and close prices of a previous period (usually the previous day for day trading).
        - **Pivot Point (P)**: The average of the high, low, and close of the previous trading period. 
        - **First Resistance (R1)**: Calculated by doubling the pivot point and then subtracting the previous low.
        - **First Support (S1)**: Derived by doubling the pivot point and then subtracting the previous high.
        - **Second Resistance (R2)**: Obtained by adding the difference of high and low (the range) to the pivot point.
        - **Second Support (S2)**: Found by subtracting the range from the pivot point.
        """)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Close'], mode='lines', name='Close Price'))
        fig1.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['Pivot'], mode='lines', name='Pivot', line=dict(dash='dash', color='black')))
        fig1.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['R1'], mode='lines', name='Resistance 1', line=dict(dash='dash', color='red')))
        fig1.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['S1'], mode='lines', name='Support 1', line=dict(dash='dash', color='green')))
        fig1.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['R2'], mode='lines', name='Resistance 2', line=dict(dash='dash', color='orange')))
        fig1.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot['S2'], mode='lines', name='Support 2', line=dict(dash='dash', color='blue')))
        fig1.update_layout(
            title=f'{ticker} Price with Pivot Points and Support/Resistance Levels',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            width=1200,
            height=600
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Plot Support and Resistance Levels using Rolling Midpoint Range
        st.write("### Rolling Midpoint Range")
        st.markdown("""
        **Support and Resistance Levels** This method uses a rolling window to identify these levels. This provides a dynamic approach to pinpointing key price levels.
        - **Support Level**: Calculated as the rolling minimum price over the specified window period. It acts as a floor where buying interest is strong enough to prevent further price declines.
        - **Resistance Level**: Calculated as the rolling maximum price over the specified window period. It acts as a ceiling where selling interest prevents the price from rising further.
        In this analysis, the support and resistance levels are determined using a rolling window approach. Significant breaks above resistance and below support are highlighted, especially when accompanied by higher-than-average trading volumes, which could indicate potential breakout or breakdown scenarios.
        """)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price'))
        fig2.add_trace(go.Scatter(x=data.index, y=support, mode='lines', name='Support', line=dict(dash='dash', color='green')))
        fig2.add_trace(go.Scatter(x=data.index, y=resistance, mode='lines', name='Resistance', line=dict(dash='dash', color='red')))
        fig2.add_trace(go.Scatter(x=data[breaks_above_resistance].index, y=data['Close'][breaks_above_resistance], mode='markers', name='Break Above Resistance', marker=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=data[breaks_below_support].index, y=data['Close'][breaks_below_support], mode='markers', name='Break Below Support', marker=dict(color='purple')))
        fig2.update_layout(
            title=f'{ticker} Price with Support and Resistance Levels',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            width=1200,
            height=600
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Plot Swing Highs and Lows
        st.write("### Swing Highs and Lows")
        st.markdown("""
        **Swing Highs and Lows** are the highest and lowest points in the price action over a specified period.
        - **Swing High**: A peak where the price is higher than the surrounding prices.
        - **Swing Low**: A trough where the price is lower than the surrounding prices.
        """)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data_with_trendlines.index, y=data_with_trendlines['Close'], mode='lines', name='Close Price'))
        fig3.add_trace(go.Scatter(x=data_with_trendlines.index, y=data_with_trendlines['Swing_High'], mode='markers', name='Swing Highs', marker=dict(color='red')))
        fig3.add_trace(go.Scatter(x=data_with_trendlines.index, y=data_with_trendlines['Swing_Low'], mode='markers', name='Swing Lows', marker=dict(color='green')))
        fig3.update_layout(
            title=f'{ticker} with Swing Highs & Lows',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            width=1200,
            height=600
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Plot Fibonacci Retracement Levels
        st.write("### Fibonacci Retracement Levels")
        st.markdown("""
        **Fibonacci Retracement Levels** are horizontal lines that indicate where support and resistance are likely to occur. They are based on Fibonacci numbers and are used to predict the future movement of asset prices. 
        - **Levels**: 23.6%, 38.2%, 50%, 61.8%, and 78.6% represent key points where the price could potentially reverse.
        """)
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price'))
        color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        for i, (level, color) in enumerate(zip(levels, color_list)):
            fig4.add_trace(go.Scatter(x=data.index, y=fib_levels[:, i], mode='lines', name=f'Fib {level:.3f}', line=dict(color=color)))
        fig4.update_layout(
            title=f'{ticker} with Fibonacci Retracement Levels',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            width=1200,
            height=600
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Plot Trendlines
        st.write("### Trendlines with Regression Analysis")
        st.markdown("""
        **Trendlines** are straight lines drawn on a price chart to connect two or more price points. They help identify the direction of the market trend and potential areas of support and resistance. In this analysis, trendlines are determined using regression analysis to fit the lines through swing highs and lows.
        - **Upper Trendline**: Connects higher highs using linear regression to fit a line through these points. This line acts as a resistance level.
        - **Lower Trendline**: Connects lower lows using linear regression to fit a line through these points. This line acts as a support level.
        1. **Swing Highs and Lows Identification**: First, local maxima (swing highs) and minima (swing lows) are identified using a specified lookback period.
        2. **Linear Regression**: A linear regression is then applied to the swing highs to form the upper trendline and to the swing lows to form the lower trendline. 
        3. **Visualization**: The trendlines are plotted along with the stock's closing prices to represent of potential resistance and support levels.
        """)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=data_with_trendlines.index, y=data_with_trendlines['Close'], mode='lines', name='Close Price'))
        
        swing_highs = data_with_trendlines['Swing_High'].dropna()
        swing_lows = data_with_trendlines['Swing_Low'].dropna()

        if len(swing_highs) > 1 and len(swing_lows) > 1:
            upper_m, upper_b = np.polyfit(swing_highs.index.map(pd.Timestamp.toordinal), swing_highs.values, 1)
            lower_m, lower_b = np.polyfit(swing_lows.index.map(pd.Timestamp.toordinal), swing_lows.values, 1)
            data_with_trendlines['Upper_Trendline'] = upper_m * data_with_trendlines.index.map(pd.Timestamp.toordinal) + upper_b
            data_with_trendlines['Lower_Trendline'] = lower_m * data_with_trendlines.index.map(pd.Timestamp.toordinal) + lower_b
            fig5.add_trace(go.Scatter(x=data_with_trendlines.index, y=data_with_trendlines['Upper_Trendline'], mode='lines', name='Upper Trendline', line=dict(color='orange')))
            fig5.add_trace(go.Scatter(x=data_with_trendlines.index, y=data_with_trendlines['Lower_Trendline'], mode='lines', name='Lower Trendline', line=dict(color='blue')))
        
        fig5.update_layout(
            title=f'{ticker} with Trendlines',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            width=1200,
            height=600
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Plot Volume Profile
        st.write("### Volume Profile")
        st.markdown("""
        **Volume Profile** is a charting tool that shows the amount of volume traded at different price levels over a specified period. It helps identify areas of high trading activity, which can act as support or resistance. 
        - **High Volume Areas**: Indicate significant trading activity and can act as strong support or resistance levels.
        """)
        
        fig6, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 1]})
        ax1.plot(data['Close'], label="Close Price")
        current_price = data['Close'].iloc[-1]
        support_idx = np.argmax(volume_profile[:np.digitize(current_price, price_bins)])
        resistance_idx = np.argmax(volume_profile[np.digitize(current_price, price_bins):]) + np.digitize(current_price, price_bins)
        support_price = price_bins[support_idx]
        resistance_price = price_bins[resistance_idx]
        ax1.axhline(y=support_price, color='g', linestyle='--', label='Support')
        ax1.axhline(y=resistance_price, color='r', linestyle='--', label='Resistance')
        ax1.annotate(f'Support: {support_price:.2f}', 
                     xy=(data.index[-1], support_price), 
                     xytext=(data.index[-1], support_price - 5), 
                     arrowprops=dict(facecolor='green', arrowstyle='->'),
                     color='green', fontsize=12)
        ax1.annotate(f'Resistance: {resistance_price:.2f}', 
                     xy=(data.index[-1], resistance_price), 
                     xytext=(data.index[-1], resistance_price + 5), 
                     arrowprops=dict(facecolor='red', arrowstyle='->'),
                     color='red', fontsize=12)
        ax1.legend()
        ax1.set_title(f'{ticker} Price Data')
        ax2.barh(price_bins[:-1], volume_profile, height=(price_bins[1] - price_bins[0]), color='blue', edgecolor='none')
        ax2.set_title('Volume Profile')
        st.pyplot(fig6, use_container_width=True)

        # Plot KMeans Clusters
        st.write("### KMeans Clusters")
        st.markdown("""
        **KMeans Clustering** is a machine learning algorithm used to partition a dataset into clusters. In the context of stock prices, it helps identify patterns and group similar price movements together.
        - **Clusters**: Represent different regimes or phases in the stock price movements.
        """)
        
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        for center in cluster_centers:
            fig7.add_trace(go.Scatter(x=[data.index[-1]], y=[center], mode='markers+text', name=f'Cluster Center: {center:.2f}', text=[f'{center:.2f}'], textposition='top center'))
            fig7.add_shape(type="line",
                           x0=data.index[0], x1=data.index[-1], y0=center, y1=center,
                           line=dict(color='Red', dash="dash"))
        fig7.update_layout(
            title=f'{ticker} with KMeans Clustering (Last {n_days} Days)',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            width=1200,
            height=600
        )
        st.plotly_chart(fig7, use_container_width=True)

    else:
        st.write("No data found for the given ticker and date range.")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)