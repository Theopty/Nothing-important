#!/usr/bin/env python3
"""
Interactive Data Viewer - Sort and filter analysis data
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta
from src.storage.database import NewsStockDatabase


def view_articles(db, sort_by='score', sentiment_filter=None, source_filter=None, days=30):
    """View and sort articles"""

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    articles = db.get_articles_by_date(
        start_date.isoformat(),
        end_date.isoformat()
    )

    if not articles:
        print("No articles found. Run 'python main.py' first.")
        return

    # Convert to DataFrame for easy sorting
    df = pd.DataFrame(articles)

    # Apply filters
    if sentiment_filter:
        df = df[df['sentiment_type'] == sentiment_filter.lower()]

    if source_filter:
        df = df[df['source'].str.contains(source_filter, case=False, na=False)]

    # Sort
    sort_columns = {
        'score': 'total_score',
        'sentiment': 'sentiment_polarity',
        'date': 'publish_date',
        'source': 'source'
    }

    if sort_by in sort_columns:
        df = df.sort_values(by=sort_columns[sort_by], ascending=False)

    # Display
    print(f"\n{'='*80}")
    print(f"ARTICLES (Sorted by: {sort_by.upper()}, Filter: {sentiment_filter or 'All'}, Source: {source_filter or 'All'})")
    print(f"{'='*80}")
    print(f"Found {len(df)} articles\n")

    for idx, row in df.iterrows():
        sentiment_emoji = '🟢' if row['sentiment_type'] == 'positive' else '🔴' if row['sentiment_type'] == 'negative' else '⚪'

        print(f"{sentiment_emoji} [{row['source']}] {row['title']}")
        print(f"   Score: {row['total_score']:.3f} | Sentiment: {row['sentiment_type']} ({row['sentiment_polarity']:+.2f})")
        print(f"   Date: {row['publish_date']}")
        print()


def view_by_source(db, days=30):
    """View articles grouped by source"""

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    articles = db.get_articles_by_date(
        start_date.isoformat(),
        end_date.isoformat()
    )

    if not articles:
        print("No articles found.")
        return

    df = pd.DataFrame(articles)

    print(f"\n{'='*80}")
    print(f"ARTICLES BY SOURCE")
    print(f"{'='*80}\n")

    # Group by source
    source_stats = df.groupby('source').agg({
        'title': 'count',
        'sentiment_polarity': 'mean',
        'total_score': 'mean'
    }).sort_values('title', ascending=False)

    source_stats.columns = ['Count', 'Avg Sentiment', 'Avg Score']

    print(source_stats.to_string())

    # Sentiment distribution by source
    print(f"\n{'='*80}")
    print(f"SENTIMENT DISTRIBUTION BY SOURCE")
    print(f"{'='*80}\n")

    sentiment_by_source = pd.crosstab(df['source'], df['sentiment_type'])
    print(sentiment_by_source.to_string())


def view_by_stock(db, days=30):
    """View data grouped by stock symbol"""

    print(f"\n{'='*80}")
    print(f"ANALYSIS BY STOCK SYMBOL")
    print(f"{'='*80}\n")

    cursor = db.conn.cursor()

    # Get daily aggregates
    cursor.execute("""
        SELECT symbol,
               COUNT(*) as days,
               SUM(article_count) as total_articles,
               AVG(average_sentiment) as avg_sentiment,
               SUM(positive_count) as total_positive,
               SUM(negative_count) as total_negative,
               SUM(neutral_count) as total_neutral
        FROM daily_aggregates
        GROUP BY symbol
        ORDER BY total_articles DESC
    """)

    rows = cursor.fetchall()

    if not rows:
        print("No stock data found. Run 'python main.py --symbols AAPL MSFT' first.")
        return

    # Create DataFrame
    df = pd.DataFrame([dict(row) for row in rows])

    print(df.to_string(index=False))

    # Get correlations
    print(f"\n{'='*80}")
    print(f"CORRELATIONS BY STOCK")
    print(f"{'='*80}\n")

    cursor.execute("""
        SELECT symbol,
               sentiment_price_corr,
               predictive_corr,
               significant_days_count,
               days_analyzed
        FROM correlations
        ORDER BY ABS(predictive_corr) DESC
    """)

    corr_rows = cursor.fetchall()

    if corr_rows:
        corr_df = pd.DataFrame([dict(row) for row in corr_rows])
        corr_df.columns = ['Symbol', 'Sentiment↔Price', 'Predictive', 'Sig Days', 'Days']
        print(corr_df.to_string(index=False))
    else:
        print("No correlation data found.")


def view_by_sentiment(db, days=30):
    """View articles grouped by sentiment"""

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    articles = db.get_articles_by_date(
        start_date.isoformat(),
        end_date.isoformat()
    )

    if not articles:
        print("No articles found.")
        return

    df = pd.DataFrame(articles)

    print(f"\n{'='*80}")
    print(f"ARTICLES BY SENTIMENT")
    print(f"{'='*80}\n")

    # Overall stats
    sentiment_counts = df['sentiment_type'].value_counts()
    print("Distribution:")
    for sentiment, count in sentiment_counts.items():
        emoji = '🟢' if sentiment == 'positive' else '🔴' if sentiment == 'negative' else '⚪'
        print(f"  {emoji} {sentiment.capitalize()}: {count} ({count/len(df)*100:.1f}%)")

    # Top positive articles
    print(f"\n{'='*80}")
    print(f"TOP POSITIVE ARTICLES")
    print(f"{'='*80}\n")

    positive = df[df['sentiment_type'] == 'positive'].sort_values('sentiment_polarity', ascending=False).head(5)

    for idx, row in positive.iterrows():
        print(f"🟢 {row['title']}")
        print(f"   Sentiment: +{row['sentiment_polarity']:.3f} | Score: {row['total_score']:.3f}")
        print(f"   Source: {row['source']} | Date: {row['publish_date']}")
        print()

    # Top negative articles
    print(f"\n{'='*80}")
    print(f"TOP NEGATIVE ARTICLES")
    print(f"{'='*80}\n")

    negative = df[df['sentiment_type'] == 'negative'].sort_values('sentiment_polarity').head(5)

    for idx, row in negative.iterrows():
        print(f"🔴 {row['title']}")
        print(f"   Sentiment: {row['sentiment_polarity']:.3f} | Score: {row['total_score']:.3f}")
        print(f"   Source: {row['source']} | Date: {row['publish_date']}")
        print()


def main():
    parser = argparse.ArgumentParser(description='View and sort analysis data')
    parser.add_argument('--view', choices=['articles', 'sources', 'stocks', 'sentiment'],
                       default='articles', help='What to view')
    parser.add_argument('--sort', choices=['score', 'sentiment', 'date', 'source'],
                       default='score', help='Sort articles by')
    parser.add_argument('--sentiment', choices=['positive', 'negative', 'neutral'],
                       help='Filter by sentiment')
    parser.add_argument('--source', help='Filter by source name (partial match)')
    parser.add_argument('--days', type=int, default=30, help='Days of history')
    parser.add_argument('--db', default='data/news_stock_analysis.db', help='Database path')

    args = parser.parse_args()

    db = NewsStockDatabase(args.db)

    if args.view == 'articles':
        view_articles(db, sort_by=args.sort, sentiment_filter=args.sentiment,
                     source_filter=args.source, days=args.days)
    elif args.view == 'sources':
        view_by_source(db, days=args.days)
    elif args.view == 'stocks':
        view_by_stock(db, days=args.days)
    elif args.view == 'sentiment':
        view_by_sentiment(db, days=args.days)

    db.close()


if __name__ == "__main__":
    main()
