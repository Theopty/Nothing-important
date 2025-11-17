#!/usr/bin/env python3
"""
Source Impact Analyzer - Compare how different news sources affect stock prices
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.storage.database import NewsStockDatabase
from src.utils.visualizer import COLORS


def analyze_source_impact(db, symbol: str, days: int = 30):
    """
    Analyze which news sources have the most impact on stock prices

    Args:
        db: Database instance
        symbol: Stock ticker
        days: Number of days to analyze
    """
    print(f"\n{'='*80}")
    print(f"SOURCE IMPACT ANALYSIS FOR {symbol}")
    print(f"{'='*80}\n")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Get articles
    articles = db.get_articles_by_date(
        start_date.isoformat(),
        end_date.isoformat()
    )

    if not articles:
        print("No articles found. Run analysis first.")
        return

    # Get stock data
    stock_df = db.get_stock_data(
        symbol,
        start_date.date().isoformat(),
        end_date.date().isoformat()
    )

    if stock_df.empty:
        print(f"No stock data for {symbol}")
        return

    # Normalize column names (database uses lowercase)
    stock_df.columns = stock_df.columns.str.capitalize()

    # Calculate derived metrics (these aren't stored in database)
    stock_df['Daily_Return'] = stock_df['Close'].pct_change() * 100  # Convert to percentage
    stock_df['Volume_Change'] = stock_df['Volume'].pct_change()

    # Convert to DataFrame
    df_articles = pd.DataFrame(articles)
    df_articles['publish_date'] = pd.to_datetime(df_articles['publish_date'])
    df_articles['date'] = df_articles['publish_date'].dt.date

    # Get unique sources
    sources = df_articles['source'].value_counts()

    print(f"Analyzing {len(sources)} news sources over {days} days")
    print(f"Total articles: {len(df_articles)}\n")

    # Analyze each source
    source_impacts = []

    for source in sources.index[:10]:  # Top 10 sources
        source_articles = df_articles[df_articles['source'] == source]

        # Group by date
        daily_source = source_articles.groupby('date').agg({
            'sentiment_polarity': 'mean',
            'total_score': 'mean',
            'title': 'count'
        }).reset_index()

        daily_source.columns = ['date', 'avg_sentiment', 'avg_score', 'article_count']
        daily_source['date'] = pd.to_datetime(daily_source['date']).dt.tz_localize(None)
        daily_source.set_index('date', inplace=True)

        # Merge with stock data - ensure timezone compatibility
        stock_df.index = pd.to_datetime(stock_df.index).tz_localize(None)
        merged = daily_source.join(stock_df[['Close', 'Daily_Return', 'Volume']], how='inner')

        if len(merged) < 3:
            continue

        # Calculate correlations
        sentiment_price_corr = merged['avg_sentiment'].corr(merged['Daily_Return'])
        score_price_corr = merged['avg_score'].corr(merged['Daily_Return'])

        # Calculate next-day impact (predictive)
        merged['next_day_return'] = merged['Daily_Return'].shift(-1)
        predictive_corr = merged['avg_sentiment'].corr(merged['next_day_return'])

        # Separate positive and negative news impact
        positive_news = merged[merged['avg_sentiment'] > 0]
        negative_news = merged[merged['avg_sentiment'] < 0]

        pos_impact = positive_news['avg_sentiment'].corr(positive_news['Daily_Return']) if len(positive_news) > 2 else 0
        neg_impact = negative_news['avg_sentiment'].corr(negative_news['Daily_Return']) if len(negative_news) > 2 else 0

        # Calculate average price change after news
        avg_return_after_positive = positive_news['next_day_return'].mean() if len(positive_news) > 0 else 0
        avg_return_after_negative = negative_news['next_day_return'].mean() if len(negative_news) > 0 else 0

        source_impacts.append({
            'source': source,
            'article_count': len(source_articles),
            'days_with_news': len(merged),
            'avg_sentiment': source_articles['sentiment_polarity'].mean(),
            'sentiment_price_corr': sentiment_price_corr,
            'predictive_corr': predictive_corr,
            'positive_news_impact': pos_impact,
            'negative_news_impact': neg_impact,
            'avg_return_after_positive': avg_return_after_positive * 100,  # Convert to %
            'avg_return_after_negative': avg_return_after_negative * 100,
            'positive_count': len(positive_news),
            'negative_count': len(negative_news)
        })

    # Create DataFrame and sort by predictive power
    impact_df = pd.DataFrame(source_impacts)
    impact_df = impact_df.sort_values('predictive_corr', key=abs, ascending=False)

    # Display results
    print(f"{'='*80}")
    print(f"SOURCE IMPACT RANKING (by predictive power)")
    print(f"{'='*80}\n")

    print(f"{'Source':<20} {'Articles':<10} {'Predict':<10} {'Pos Impact':<12} {'Neg Impact':<12}")
    print("-" * 80)

    for idx, row in impact_df.iterrows():
        predict_str = f"{row['predictive_corr']:+.3f}"
        pos_str = f"{row['positive_news_impact']:+.3f}"
        neg_str = f"{row['negative_news_impact']:+.3f}"

        print(f"{row['source']:<20} {row['article_count']:<10} {predict_str:<10} {pos_str:<12} {neg_str:<12}")

    # Detailed analysis
    print(f"\n{'='*80}")
    print(f"DETAILED SOURCE ANALYSIS")
    print(f"{'='*80}\n")

    for idx, row in impact_df.head(5).iterrows():
        print(f"📰 {row['source']}")
        print(f"   Articles: {row['article_count']} ({row['positive_count']} positive, {row['negative_count']} negative)")
        print(f"   Predictive Power: {row['predictive_corr']:+.3f}")
        print(f"   Positive News Impact: {row['positive_news_impact']:+.3f}")
        print(f"   Negative News Impact: {row['negative_news_impact']:+.3f}")
        print(f"   Avg Return After Positive News: {row['avg_return_after_positive']:+.2f}%")
        print(f"   Avg Return After Negative News: {row['avg_return_after_negative']:+.2f}%")

        # Determine which type of news from this source matters more
        if abs(row['negative_news_impact']) > abs(row['positive_news_impact']):
            impact_type = "NEGATIVE"
            impact_diff = abs(row['negative_news_impact']) - abs(row['positive_news_impact'])
            print(f"   ⚠️  This source's NEGATIVE news has {impact_diff:.3f} more impact!")
        else:
            impact_type = "POSITIVE"
            impact_diff = abs(row['positive_news_impact']) - abs(row['negative_news_impact'])
            print(f"   ✅ This source's POSITIVE news has {impact_diff:.3f} more impact!")

        print()

    # Find the most influential sources
    print(f"{'='*80}")
    print(f"KEY INSIGHTS")
    print(f"{'='*80}\n")

    # Most predictive source overall
    most_predictive = impact_df.iloc[0]
    print(f"🏆 Most Predictive Source: {most_predictive['source']}")
    print(f"   Predictive Correlation: {most_predictive['predictive_corr']:+.3f}")

    # Source with strongest negative impact
    strongest_negative = impact_df.loc[impact_df['negative_news_impact'].abs().idxmax()]
    print(f"\n🔴 Strongest Negative News Impact: {strongest_negative['source']}")
    print(f"   Negative Impact Score: {strongest_negative['negative_news_impact']:+.3f}")
    print(f"   Avg Return After Negative News: {strongest_negative['avg_return_after_negative']:+.2f}%")

    # Source with strongest positive impact
    strongest_positive = impact_df.loc[impact_df['positive_news_impact'].abs().idxmax()]
    print(f"\n🟢 Strongest Positive News Impact: {strongest_positive['source']}")
    print(f"   Positive Impact Score: {strongest_positive['positive_news_impact']:+.3f}")
    print(f"   Avg Return After Positive News: {strongest_positive['avg_return_after_positive']:+.2f}%")

    # Generate visualizations
    generate_source_impact_charts(impact_df, symbol)

    return impact_df


def generate_source_impact_charts(impact_df, symbol):
    """Generate charts comparing source impacts"""

    output_dir = Path('reports/charts')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set dark theme
    plt.style.use('dark_background')

    # 1. Predictive Power Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sort by predictive power
    top_sources = impact_df.head(10).sort_values('predictive_corr')

    # Bar chart - Predictive Power
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in top_sources['predictive_corr']]
    ax1.barh(range(len(top_sources)), top_sources['predictive_corr'], color=colors, alpha=0.8, edgecolor='white')
    ax1.set_yticks(range(len(top_sources)))
    ax1.set_yticklabels(top_sources['source'], fontsize=10)
    ax1.set_xlabel('Predictive Correlation', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} - Source Predictive Power', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color=COLORS['neutral'], linestyle='--', linewidth=2)
    ax1.grid(True, alpha=0.3, axis='x')

    # Scatter plot - Positive vs Negative Impact
    ax2.scatter(impact_df['positive_news_impact'], impact_df['negative_news_impact'],
               s=impact_df['article_count'] * 10, alpha=0.6, color=COLORS['accent'],
               edgecolors='white', linewidth=2)

    # Add source labels
    for idx, row in impact_df.head(5).iterrows():
        ax2.annotate(row['source'], (row['positive_news_impact'], row['negative_news_impact']),
                    fontsize=9, alpha=0.8, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['primary'], alpha=0.3))

    ax2.set_xlabel('Positive News Impact', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Negative News Impact', fontsize=12, fontweight='bold')
    ax2.set_title(f'{symbol} - Positive vs Negative News Impact by Source', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color=COLORS['neutral'], linestyle='--', linewidth=2)
    ax2.axvline(x=0, color=COLORS['neutral'], linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = output_dir / f'{symbol}_source_impact.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"\n✓ Created {filepath}")
    plt.close()

    # 2. Average Returns After News
    fig, ax = plt.subplots(figsize=(14, 8))

    top_sources = impact_df.head(10).sort_values('avg_return_after_negative')

    x = np.arange(len(top_sources))
    width = 0.35

    ax.barh(x - width/2, top_sources['avg_return_after_positive'],
           width, label='After Positive News', color=COLORS['positive'], alpha=0.8, edgecolor='white')
    ax.barh(x + width/2, top_sources['avg_return_after_negative'],
           width, label='After Negative News', color=COLORS['negative'], alpha=0.8, edgecolor='white')

    ax.set_yticks(x)
    ax.set_yticklabels(top_sources['source'], fontsize=10)
    ax.set_xlabel('Average Next-Day Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{symbol} - Stock Returns After News by Source', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    filepath = output_dir / f'{symbol}_source_returns.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✓ Created {filepath}")
    plt.close()


def compare_sources_head_to_head(db, symbol: str, source1: str, source2: str, days: int = 30):
    """
    Compare two specific sources head-to-head

    Args:
        db: Database instance
        symbol: Stock ticker
        source1: First source name
        source2: Second source name
        days: Number of days
    """
    print(f"\n{'='*80}")
    print(f"HEAD-TO-HEAD: {source1} vs {source2}")
    print(f"{'='*80}\n")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    articles = db.get_articles_by_date(
        start_date.isoformat(),
        end_date.isoformat()
    )

    df = pd.DataFrame(articles)

    # Filter for each source
    s1_articles = df[df['source'].str.contains(source1, case=False, na=False)]
    s2_articles = df[df['source'].str.contains(source2, case=False, na=False)]

    print(f"{source1}:")
    print(f"  Articles: {len(s1_articles)}")
    print(f"  Avg Sentiment: {s1_articles['sentiment_polarity'].mean():+.3f}")
    print(f"  Positive: {len(s1_articles[s1_articles['sentiment_type'] == 'positive'])}")
    print(f"  Negative: {len(s1_articles[s1_articles['sentiment_type'] == 'negative'])}")

    print(f"\n{source2}:")
    print(f"  Articles: {len(s2_articles)}")
    print(f"  Avg Sentiment: {s2_articles['sentiment_polarity'].mean():+.3f}")
    print(f"  Positive: {len(s2_articles[s2_articles['sentiment_type'] == 'positive'])}")
    print(f"  Negative: {len(s2_articles[s2_articles['sentiment_type'] == 'negative'])}")


def main():
    parser = argparse.ArgumentParser(description='Analyze news source impact on stock prices')
    parser.add_argument('symbol', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--compare', nargs=2, metavar=('SOURCE1', 'SOURCE2'),
                       help='Compare two sources head-to-head')
    parser.add_argument('--db', default='data/news_stock_analysis.db', help='Database path')

    args = parser.parse_args()

    db = NewsStockDatabase(args.db)

    if args.compare:
        compare_sources_head_to_head(db, args.symbol.upper(), args.compare[0], args.compare[1], args.days)
    else:
        analyze_source_impact(db, args.symbol.upper(), args.days)

    db.close()


if __name__ == "__main__":
    main()
