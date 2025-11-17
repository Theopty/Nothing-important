"""
Stock Data Collector - Fetches stock market data from various sources
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockCollector:
    """Collects stock market data using yfinance"""

    def __init__(self):
        self.cache = {}

    def fetch_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a specific symbol

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1m, 5m, 1h, 1d, 1wk, 1mo)

        Returns:
            DataFrame with stock data (Open, High, Low, Close, Volume)
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Add symbol column
            df['Symbol'] = symbol

            # Calculate additional metrics
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()

            # Calculate moving averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()

            # Calculate volatility (standard deviation of returns)
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

            logger.info(f"Fetched {len(df)} data points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None

    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}

        for symbol in symbols:
            df = self.fetch_stock_data(symbol, start_date, end_date, interval)
            if df is not None:
                results[symbol] = df

        return results

    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a stock

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', ''),
                'employees': info.get('fullTimeEmployees', 0)
            }

        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def get_sector_etf_data(
        self,
        sector_etfs: Dict[str, str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch sector ETF data for industry analysis

        Args:
            sector_etfs: Dictionary mapping sector name to ETF symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping sector to DataFrame
        """
        results = {}

        for sector, etf_symbol in sector_etfs.items():
            df = self.fetch_stock_data(etf_symbol, start_date, end_date)
            if df is not None:
                results[sector] = df

        return results

    def detect_volume_spikes(
        self,
        df: pd.DataFrame,
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect volume spikes in stock data

        Args:
            df: Stock DataFrame
            threshold: Spike threshold (1.5 = 150% of average)

        Returns:
            DataFrame with spike indicators
        """
        # Calculate average volume (20-day)
        df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()

        # Detect spikes
        df['Volume_Spike'] = df['Volume'] > (df['Avg_Volume_20'] * threshold)

        # Calculate spike magnitude
        df['Spike_Magnitude'] = df['Volume'] / df['Avg_Volume_20']

        return df

    def calculate_correlation_metrics(
        self,
        df: pd.DataFrame,
        news_timestamp: datetime,
        window_hours: int = 24
    ) -> Dict:
        """
        Calculate metrics for correlating with news

        Args:
            df: Stock DataFrame
            news_timestamp: Timestamp of news event
            window_hours: Time window to analyze

        Returns:
            Dictionary with correlation metrics
        """
        try:
            # Find closest data point to news timestamp
            df_reset = df.reset_index()
            df_reset['TimeDiff'] = abs(df_reset['Date'] - news_timestamp)
            closest_idx = df_reset['TimeDiff'].idxmin()

            # Get data points around news time
            start_idx = max(0, closest_idx - 3)
            end_idx = min(len(df_reset), closest_idx + 3)

            window_data = df_reset.iloc[start_idx:end_idx]

            # Calculate metrics
            pre_news_price = window_data.iloc[0]['Close'] if len(window_data) > 0 else 0
            post_news_price = window_data.iloc[-1]['Close'] if len(window_data) > 0 else 0

            price_change = ((post_news_price - pre_news_price) / pre_news_price * 100) if pre_news_price > 0 else 0

            avg_volume_before = window_data.iloc[:len(window_data)//2]['Volume'].mean() if len(window_data) > 1 else 0
            avg_volume_after = window_data.iloc[len(window_data)//2:]['Volume'].mean() if len(window_data) > 1 else 0

            volume_change = ((avg_volume_after - avg_volume_before) / avg_volume_before * 100) if avg_volume_before > 0 else 0

            return {
                'news_timestamp': news_timestamp,
                'price_change_percent': price_change,
                'volume_change_percent': volume_change,
                'pre_news_price': pre_news_price,
                'post_news_price': post_news_price,
                'volatility_spike': window_data['Volatility'].max() if 'Volatility' in window_data else 0
            }

        except Exception as e:
            logger.error(f"Error calculating correlation metrics: {e}")
            return {}

    def get_intraday_data(
        self,
        symbol: str,
        date: datetime,
        interval: str = '5m'
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday data for precise correlation with news timing

        Args:
            symbol: Stock symbol
            date: Specific date
            interval: Interval (1m, 5m, 15m, 30m, 1h)

        Returns:
            DataFrame with intraday data
        """
        try:
            ticker = yf.Ticker(symbol)

            # yfinance requires period or start/end for intraday
            # For a specific day, we use period='1d' and filter
            df = ticker.history(period='5d', interval=interval)

            if df.empty:
                return None

            # Filter to specific date
            df_reset = df.reset_index()
            df_reset['Date_Only'] = df_reset['Datetime'].dt.date
            target_date = date.date()

            df_filtered = df_reset[df_reset['Date_Only'] == target_date]

            if df_filtered.empty:
                logger.warning(f"No intraday data for {symbol} on {target_date}")
                return None

            return df_filtered.set_index('Datetime')

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None


if __name__ == "__main__":
    # Test the collector
    collector = StockCollector()

    # Test fetching stock data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    df = collector.fetch_stock_data('AAPL', start_date, end_date)

    if df is not None:
        print(f"Fetched {len(df)} data points for AAPL")
        print(f"\nRecent data:")
        print(df.tail())

        # Test stock info
        info = collector.get_stock_info('AAPL')
        print(f"\nStock Info:")
        print(f"Name: {info.get('name')}")
        print(f"Sector: {info.get('sector')}")
        print(f"Industry: {info.get('industry')}")
