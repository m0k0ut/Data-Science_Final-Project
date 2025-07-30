# 📈 Stock Support and Resistance Analysis Tool

A comprehensive Streamlit web application for identifying key support and resistance price levels in financial instruments using advanced algorithmic methods and machine learning techniques.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.36.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 📊 Multiple Analysis Methods

- **Pivot Points**: Short-term trend indicators for support/resistance levels
- **Rolling Midpoint Range**: Dynamic support and resistance using rolling windows
- **Swing Highs and Lows**: Local extrema identification for trend analysis
- **Fibonacci Retracement**: Key reversal levels based on Fibonacci ratios
- **Trendlines**: Regression-based trend identification
- **Volume Profile**: Price level analysis based on trading volume
- **K-Means Clustering**: Machine learning approach to identify price patterns

### 🔄 Robust Data Handling

- **Multi-source fallback**: 4 different download strategies
- **Smart caching**: 5-minute data caching for improved performance
- **Rate limiting protection**: Automatic retry with exponential backoff
- **Error recovery**: Comprehensive error handling and user guidance

### 📱 User-Friendly Interface

- **Interactive sidebar**: Easy parameter adjustment
- **Quick examples**: One-click popular ticker selection
- **Real-time feedback**: Progress indicators and status messages
- **Responsive design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/stock-analysis-tool.git
   cd stock-analysis-tool
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

## 📋 Usage Guide

### Basic Usage

1. **Enter a Ticker Symbol**

   - Stocks: `AAPL`, `GOOGL`, `MSFT`, `TSLA`
   - ETFs: `SPY`, `QQQ`, `VTI`, `IWM`
   - Crypto: `BTC-USD`, `ETH-USD`, `ADA-USD`
   - Indices: `^GSPC`, `^DJI`, `^IXIC`

2. **Set Date Range**

   - Choose start and end dates for analysis
   - Recommended: 6 months to 2 years of data

3. **Adjust Parameters** (Optional)

   - Window periods for calculations
   - Lookback periods for trend analysis
   - Number of clusters for K-means

4. **Run Analysis**
   - Click "Run Analysis" button
   - Wait for data download and processing
   - View interactive charts and insights

### Quick Examples

Use the sidebar quick examples for instant setup:

- 📈 **AAPL** - Apple Inc. stock analysis
- 🏦 **SPY** - S&P 500 ETF analysis
- ₿ **BTC-USD** - Bitcoin cryptocurrency analysis
- 📊 **^GSPC** - S&P 500 Index analysis

## 🛠️ Technical Details

### Data Source

- **Primary**: Yahoo Finance via `yfinance` library
- **Backup methods**: Multiple fallback strategies for reliability
- **Update frequency**: Real-time market data
- **Historical range**: Up to 10+ years depending on instrument

### Analysis Methods

#### 1. Pivot Points

- **Purpose**: Identify intraday support/resistance levels
- **Calculation**: Based on previous period's High, Low, Close
- **Levels**: Pivot, R1, R2, S1, S2
- **Best for**: Day trading and short-term analysis

#### 2. Rolling Midpoint Range

- **Purpose**: Dynamic support/resistance identification
- **Method**: Rolling window min/max calculations
- **Features**: Breakout detection with volume confirmation
- **Best for**: Swing trading and medium-term analysis

#### 3. Swing Highs and Lows

- **Purpose**: Identify trend reversal points
- **Method**: Local extrema detection using scipy
- **Parameters**: Configurable lookback period
- **Best for**: Trend analysis and pattern recognition

#### 4. Fibonacci Retracement

- **Purpose**: Predict potential reversal levels
- **Levels**: 23.6%, 38.2%, 50%, 61.8%, 78.6%
- **Calculation**: Based on recent high-low range
- **Best for**: Entry/exit point identification

#### 5. Trendlines

- **Purpose**: Identify trend direction and strength
- **Method**: Linear regression on swing points
- **Features**: Upper and lower trendline calculation
- **Best for**: Long-term trend analysis

#### 6. Volume Profile

- **Purpose**: Identify high-activity price levels
- **Method**: Volume distribution across price ranges
- **Features**: Support/resistance based on trading activity
- **Best for**: Understanding market structure

#### 7. K-Means Clustering

- **Purpose**: Pattern recognition in price movements
- **Method**: Machine learning clustering algorithm
- **Features**: Configurable number of clusters
- **Best for**: Advanced pattern analysis

### Performance Optimizations

- **Caching**: `@st.cache_data` with 5-minute TTL
- **Lazy loading**: Data downloaded only when needed
- **Efficient calculations**: Vectorized operations with pandas/numpy
- **Memory management**: Automatic cleanup of large datasets

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set custom cache directory
STREAMLIT_CACHE_DIR=/path/to/cache

# Optional: Set custom port
STREAMLIT_PORT=8501
```

### Parameter Ranges

- **Window Period**: 10-60 days (default: 30)
- **Lookback Period**: 10-60 days (default: 30)
- **Volume Profile Days**: 30-365 days (default: 60)
- **K-Means Clusters**: 2-10 clusters (default: 3)

## 🐛 Troubleshooting

### Common Issues

#### "No data found" Error

**Causes:**

- Invalid ticker symbol
- Yahoo Finance rate limiting
- Network connectivity issues

**Solutions:**

1. Wait 2-3 minutes and retry
2. Try popular tickers (AAPL, SPY, BTC-USD)
3. Shorten date range to 3-6 months
4. Check ticker format and spelling

#### Rate Limiting

**Symptoms:**

- "Too Many Requests" errors
- Empty data returns
- Slow response times

**Solutions:**

1. Wait 2-3 minutes between requests
2. Use shorter date ranges
3. Try during off-peak hours
4. Use cached data when available

#### Performance Issues

**Symptoms:**

- Slow chart rendering
- High memory usage
- Browser freezing

**Solutions:**

1. Reduce date range
2. Lower analysis parameters
3. Close other browser tabs
4. Refresh the application

### Error Messages Guide

| Error                                   | Meaning            | Solution             |
| --------------------------------------- | ------------------ | -------------------- |
| `❌ Please enter a valid ticker symbol` | Empty ticker input | Enter a valid ticker |
| `❌ Start date must be before end date` | Invalid date range | Fix date selection   |
| `⚠️ Rate limiting detected`             | Too many API calls | Wait and retry       |
| `⚠️ Missing required columns`           | Incomplete data    | Try different ticker |

## 📊 Example Analysis

### Apple Inc. (AAPL) Analysis

```
Ticker: AAPL
Date Range: 2023-01-01 to 2024-01-01
Key Findings:
- Strong support at $150 level
- Resistance around $200 level
- Fibonacci 61.8% at $175
- Volume spike at $180 level
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Include type hints where appropriate

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free financial data
- **Streamlit** for the amazing web app framework
- **Plotly** for interactive charting capabilities
- **scikit-learn** for machine learning algorithms
- **pandas** and **numpy** for data processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/stock-analysis-tool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/stock-analysis-tool/discussions)
- **Email**: support@yourproject.com

## 🔮 Roadmap

### Upcoming Features

- [ ] Additional technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Portfolio analysis capabilities
- [ ] Export functionality (PDF, CSV)
- [ ] Real-time alerts and notifications
- [ ] Mobile app version
- [ ] API integration for multiple data sources

### Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Added robust error handling and caching
- **v1.2.0** - Improved UI/UX and quick examples

---

**⭐ If you find this tool useful, please give it a star on GitHub!**

**📢 Share with fellow traders and analysts!**

---

_Disclaimer: This tool is for educational and informational purposes only. It should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions._
