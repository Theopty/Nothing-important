#!/usr/bin/env python3
"""
Quick Chart Generator - Visualize stock and sentiment data
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.storage.database import NewsStockDatabase
from src.utils.visualizer import Visualizer


def generate_charts(symbol: str, days: int = 7):
    """
    Generate charts for a specific stock symbol

    Args:
        symbol: Stock ticker (e.g., AAPL, MSFT)
        days: Number of days to visualize
    """
    print(f"Generating charts for {symbol}...")

    db = NewsStockDatabase()
    viz = Visualizer()

    # Date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    print(f"Date range: {start_date.date()} to {end_date.date()}")

    # Get daily aggregates for this symbol
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT * FROM daily_aggregates
        WHERE symbol = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """, (symbol, start_date.date().isoformat(), end_date.date().isoformat()))

    daily_agg = cursor.fetchall()

    if not daily_agg:
        print(f"❌ No sentiment data found for {symbol}")
        print(f"   Run: python main.py --symbols {symbol} --days {days}")
        return

    daily_agg_list = [dict(row) for row in daily_agg]
    print(f"✓ Found {len(daily_agg_list)} days of sentiment data")

    # Get stock data
    stock_df = db.get_stock_data(
        symbol,
        start_date.date().isoformat(),
        end_date.date().isoformat()
    )

    if stock_df.empty:
        print(f"❌ No stock price data found for {symbol}")
        return

    print(f"✓ Found {len(stock_df)} days of stock data")

    # Generate visualizations
    output_dir = Path('reports/charts')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating charts...")

    try:
        # 1. Sentiment timeline
        viz.plot_sentiment_timeline(daily_agg_list, symbol, save=True)
        print(f"✓ Created {symbol}_sentiment_timeline.png")

        # 2. Correlation analysis
        viz.plot_correlation_analysis(daily_agg_list, stock_df, symbol, save=True)
        print(f"✓ Created {symbol}_correlation_analysis.png")

        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {symbol}")
        print(f"{'='*60}")

        # Calculate stats
        total_articles = sum(day['article_count'] for day in daily_agg_list)
        avg_sentiment = sum(day['average_sentiment'] * day['article_count'] for day in daily_agg_list) / total_articles if total_articles > 0 else 0

        print(f"\nSentiment Analysis:")
        print(f"  Total Articles: {total_articles}")
        print(f"  Average Sentiment: {avg_sentiment:+.3f}")

        print(f"\nStock Performance:")
        print(f"  Starting Price: ${stock_df['Close'].iloc[0]:.2f}")
        print(f"  Ending Price: ${stock_df['Close'].iloc[-1]:.2f}")
        price_change = (stock_df['Close'].iloc[-1] - stock_df['Close'].iloc[0]) / stock_df['Close'].iloc[0] * 100
        print(f"  Change: {price_change:+.2f}%")

        # Get correlation
        cursor.execute("""
            SELECT * FROM correlations
            WHERE symbol = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (symbol,))

        corr = cursor.fetchone()
        if corr:
            print(f"\nCorrelation Metrics:")
            print(f"  Sentiment vs Price: {corr['sentiment_price_corr']:+.3f}")
            print(f"  Predictive (next-day): {corr['predictive_corr']:+.3f}")

        print(f"\n{'='*60}")
        print(f"📊 Charts saved to: reports/charts/")
        print(f"{'='*60}")
        print(f"\nView the charts:")
        print(f"  • reports/charts/{symbol}_sentiment_timeline.png")
        print(f"  • reports/charts/{symbol}_correlation_analysis.png")
        print(f"\nOpen them with:")
        print(f"  eog reports/charts/{symbol}_*.png")
        print(f"  # or")
        print(f"  firefox reports/charts/{symbol}_*.png")

    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        import traceback
        traceback.print_exc()

    db.close()


def main():
    parser = argparse.ArgumentParser(description='Generate charts for stock analysis')
    parser.add_argument('symbol', help='Stock ticker symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to visualize')

    args = parser.parse_args()

    generate_charts(args.symbol.upper(), args.days)


if __name__ == "__main__":
    main()
