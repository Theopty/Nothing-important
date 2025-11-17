#!/usr/bin/env python3
"""
Dashboard - View analysis results with charts and metrics
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.storage.database import NewsStockDatabase
from src.utils.visualizer import Visualizer


def create_dashboard(db_path: str = 'data/news_stock_analysis.db', days: int = 7):
    """
    Create an interactive dashboard showing analysis results

    Args:
        db_path: Path to database
        days: Number of days to show
    """
    print("="*60)
    print("NEWS-STOCK SENTIMENT ANALYSIS DASHBOARD")
    print("="*60)

    db = NewsStockDatabase(db_path)
    viz = Visualizer()

    # Date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    print(f"\nAnalyzing data from {start_date.date()} to {end_date.date()}")

    # 1. Get articles summary
    articles = db.get_articles_by_date(
        start_date.isoformat(),
        end_date.isoformat()
    )

    print(f"\n{'='*60}")
    print(f"ARTICLES SUMMARY")
    print(f"{'='*60}")
    print(f"Total articles analyzed: {len(articles)}")

    if articles:
        # Sentiment distribution
        sentiments = [a['sentiment_type'] for a in articles if a['sentiment_type']]
        sentiment_counts = pd.Series(sentiments).value_counts()

        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment.capitalize()}: {count} ({count/len(articles)*100:.1f}%)")

        # Average scores
        avg_sentiment = sum(a['sentiment_polarity'] for a in articles if a['sentiment_polarity']) / len(articles)
        avg_total_score = sum(a['total_score'] for a in articles if a['total_score']) / len(articles)

        print(f"\nAverage Metrics:")
        print(f"  Sentiment Polarity: {avg_sentiment:+.3f}")
        print(f"  Total Score: {avg_total_score:.3f}")

        # Top sources
        sources = [a['source'] for a in articles if a['source']]
        if sources:
            source_counts = pd.Series(sources).value_counts().head(5)
            print(f"\nTop News Sources:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} articles")

        # Recent top articles
        print(f"\n{'='*60}")
        print(f"TOP ARTICLES BY SCORE")
        print(f"{'='*60}")

        sorted_articles = sorted(
            articles,
            key=lambda x: x['total_score'] if x['total_score'] else 0,
            reverse=True
        )[:5]

        for i, article in enumerate(sorted_articles, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Score: {article['total_score']:.3f} | Sentiment: {article['sentiment_type']} ({article['sentiment_polarity']:+.2f})")
            print(f"   Source: {article['source']} | Date: {article['publish_date']}")

    # 2. Get correlations
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*60}")

    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT * FROM correlations
        ORDER BY created_at DESC
        LIMIT 10
    """)

    correlations = cursor.fetchall()

    if correlations:
        print(f"\nFound {len(correlations)} correlation analyses:\n")

        for corr in correlations:
            print(f"Symbol: {corr['symbol']}")
            print(f"  Period: {corr['period_start'][:10]} to {corr['period_end'][:10]} ({corr['days_analyzed']} days)")
            print(f"  Sentiment vs Price:    {corr['sentiment_price_corr']:+.3f}")
            print(f"  Sentiment vs Volume:   {corr['sentiment_volume_corr']:+.3f}")
            print(f"  Score vs Price:        {corr['score_price_corr']:+.3f}")
            print(f"  Predictive (next-day): {corr['predictive_corr']:+.3f}")
            print(f"  Significant Days:      {corr['significant_days_count']}")
            print()
    else:
        print("No correlation data found. Run an analysis first.")

    # 3. Daily aggregates
    print(f"{'='*60}")
    print(f"DAILY SENTIMENT TRENDS")
    print(f"{'='*60}")

    cursor.execute("""
        SELECT date, symbol, article_count, average_sentiment,
               positive_count, negative_count, neutral_count
        FROM daily_aggregates
        ORDER BY date DESC
        LIMIT 20
    """)

    daily_data = cursor.fetchall()

    if daily_data:
        print(f"\nRecent daily summaries:\n")
        print(f"{'Date':<12} {'Symbol':<8} {'Articles':<10} {'Sentiment':<12} {'Pos/Neg/Neu'}")
        print("-" * 70)

        for day in daily_data:
            sentiment_str = f"{day['average_sentiment']:+.3f}"
            dist_str = f"{day['positive_count']}/{day['negative_count']}/{day['neutral_count']}"
            print(f"{day['date']:<12} {day['symbol']:<8} {day['article_count']:<10} {sentiment_str:<12} {dist_str}")
    else:
        print("No daily aggregate data found.")

    # 4. Experiments
    print(f"\n{'='*60}")
    print(f"EXPERIMENTS")
    print(f"{'='*60}")

    experiments = db.get_experiments(limit=5)

    if experiments:
        print(f"\nRecent experiments:\n")

        for exp in experiments:
            print(f"Name: {exp['experiment_name']} (v{exp['version']})")
            print(f"  Date: {exp['created_at']}")

            metrics = exp['metrics']
            print(f"  Avg Correlation: {metrics.get('avg_correlation', 0):.3f}")
            print(f"  Avg Confidence:  {metrics.get('avg_confidence', 0):.3f}")

            signals = metrics.get('signals', {})
            print(f"  Signals: BUY={signals.get('BUY', 0)}, SELL={signals.get('SELL', 0)}, HOLD={signals.get('HOLD', 0)}")
            print()
    else:
        print("No experiments found. Run 'python experiment_runner.py' to start experimenting.")

    # 5. Generate visualizations
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    if articles and daily_data:
        try:
            # Create sentiment distribution chart
            viz.plot_sentiment_distribution(articles, save=True)
            print("✓ Created sentiment_distribution.png")
        except Exception as e:
            print(f"✗ Could not create sentiment distribution: {e}")

    if correlations:
        try:
            # Get stock data for visualization
            latest_corr = correlations[0]
            symbol = latest_corr['symbol']

            stock_df = db.get_stock_data(
                symbol,
                latest_corr['period_start'],
                latest_corr['period_end']
            )

            if not stock_df.empty:
                # Get daily aggregates for this symbol
                cursor.execute("""
                    SELECT * FROM daily_aggregates
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                """, (symbol, latest_corr['period_start'], latest_corr['period_end']))

                daily_agg = cursor.fetchall()
                daily_agg_list = [dict(row) for row in daily_agg]

                if daily_agg_list:
                    viz.plot_sentiment_timeline(daily_agg_list, symbol, save=True)
                    print(f"✓ Created {symbol}_sentiment_timeline.png")

                    viz.plot_correlation_analysis(daily_agg_list, stock_df, symbol, save=True)
                    print(f"✓ Created {symbol}_correlation_analysis.png")
        except Exception as e:
            print(f"✗ Could not create stock visualizations: {e}")

    # 6. Create HTML report
    print(f"\n{'='*60}")
    print(f"GENERATING HTML REPORT")
    print(f"{'='*60}")

    html_report = create_html_dashboard(articles, correlations, daily_data, experiments)
    report_path = Path('reports/dashboard.html')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(html_report)

    print(f"✓ Created HTML dashboard: {report_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"DASHBOARD COMPLETE!")
    print(f"{'='*60}")
    print(f"\nFiles created:")
    print(f"  • reports/dashboard.html - Interactive HTML dashboard")
    print(f"  • reports/charts/*.png - Visualization charts")
    print(f"\nOpen the dashboard in your browser:")
    print(f"  file://{report_path.absolute()}")

    db.close()


def create_html_dashboard(articles, correlations, daily_data, experiments):
    """Create HTML dashboard"""

    # Calculate stats
    total_articles = len(articles)
    sentiment_dist = {}
    if articles:
        for a in articles:
            sent = a['sentiment_type'] or 'unknown'
            sentiment_dist[sent] = sentiment_dist.get(sent, 0) + 1

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>News-Stock Analysis Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-bottom: 20px;
            }}
            .header h1 {{
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .header p {{
                color: #666;
                font-size: 1.1em;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .stat-card h3 {{
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                margin-bottom: 10px;
            }}
            .stat-card .value {{
                color: #667eea;
                font-size: 2.5em;
                font-weight: bold;
            }}
            .section {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .section h2 {{
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }}
            th {{
                background: #f8f9fa;
                color: #667eea;
                font-weight: 600;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            .positive {{ color: #28a745; font-weight: bold; }}
            .negative {{ color: #dc3545; font-weight: bold; }}
            .neutral {{ color: #6c757d; }}
            .article-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 15px;
                border-left: 4px solid #667eea;
            }}
            .article-card h4 {{
                color: #333;
                margin-bottom: 10px;
            }}
            .article-meta {{
                color: #666;
                font-size: 0.9em;
                margin-top: 10px;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .chart-container img {{
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 0.85em;
                font-weight: bold;
                margin-right: 5px;
            }}
            .badge-positive {{ background: #d4edda; color: #155724; }}
            .badge-negative {{ background: #f8d7da; color: #721c24; }}
            .badge-neutral {{ background: #e2e3e5; color: #383d41; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 News-Stock Sentiment Dashboard</h1>
                <p>Real-time analysis of news sentiment and stock correlations</p>
                <p style="font-size: 0.9em; margin-top: 5px;">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Articles</h3>
                    <div class="value">{total_articles}</div>
                </div>
                <div class="stat-card">
                    <h3>Positive</h3>
                    <div class="value positive">{sentiment_dist.get('positive', 0)}</div>
                </div>
                <div class="stat-card">
                    <h3>Negative</h3>
                    <div class="value negative">{sentiment_dist.get('negative', 0)}</div>
                </div>
                <div class="stat-card">
                    <h3>Neutral</h3>
                    <div class="value neutral">{sentiment_dist.get('neutral', 0)}</div>
                </div>
            </div>
    """

    # Correlations section
    if correlations:
        html += """
            <div class="section">
                <h2>🔗 Correlation Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Days</th>
                            <th>Sentiment ↔ Price</th>
                            <th>Predictive</th>
                            <th>Significant Days</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for corr in correlations[:10]:
            predictive_class = 'positive' if corr['predictive_corr'] > 0.3 else 'negative' if corr['predictive_corr'] < -0.3 else 'neutral'
            html += f"""
                        <tr>
                            <td><strong>{corr['symbol']}</strong></td>
                            <td>{corr['days_analyzed']}</td>
                            <td>{corr['sentiment_price_corr']:+.3f}</td>
                            <td class="{predictive_class}">{corr['predictive_corr']:+.3f}</td>
                            <td>{corr['significant_days_count']}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>
        """

    # Charts section
    html += """
            <div class="section">
                <h2>📈 Visualizations</h2>
                <div class="chart-container">
                    <h3>Sentiment Distribution</h3>
                    <img src="charts/sentiment_distribution.png" alt="Sentiment Distribution" onerror="this.style.display='none'">
                </div>
    """

    if correlations:
        symbol = correlations[0]['symbol']
        html += f"""
                <div class="chart-container">
                    <h3>{symbol} - Sentiment Timeline</h3>
                    <img src="charts/{symbol}_sentiment_timeline.png" alt="Sentiment Timeline" onerror="this.style.display='none'">
                </div>
                <div class="chart-container">
                    <h3>{symbol} - Correlation Analysis</h3>
                    <img src="charts/{symbol}_correlation_analysis.png" alt="Correlation Analysis" onerror="this.style.display='none'">
                </div>
        """

    html += """
            </div>
    """

    # Top articles
    if articles:
        sorted_articles = sorted(
            articles,
            key=lambda x: x['total_score'] if x['total_score'] else 0,
            reverse=True
        )[:10]

        html += """
            <div class="section">
                <h2>⭐ Top Articles by Score</h2>
        """

        for article in sorted_articles:
            sentiment_class = f"badge-{article['sentiment_type']}" if article['sentiment_type'] else 'badge-neutral'
            html += f"""
                <div class="article-card">
                    <h4>{article['title']}</h4>
                    <div>
                        <span class="badge {sentiment_class}">{article['sentiment_type'] or 'unknown'}</span>
                        <span class="badge badge-neutral">Score: {article['total_score']:.3f}</span>
                    </div>
                    <div class="article-meta">
                        Source: {article['source']} | Date: {article['publish_date']} | Sentiment: {article['sentiment_polarity']:+.2f}
                    </div>
                </div>
            """

        html += """
            </div>
        """

    # Footer
    html += """
            <div class="section" style="text-align: center; color: #666;">
                <p>📊 News-Stock Sentiment Analysis Algorithm</p>
                <p style="font-size: 0.9em; margin-top: 10px;">
                    Run <code>python dashboard.py</code> to refresh this dashboard
                </p>
            </div>

        </div>
    </body>
    </html>
    """

    return html


def main():
    parser = argparse.ArgumentParser(description='View analysis dashboard')
    parser.add_argument('--db', default='data/news_stock_analysis.db', help='Database path')
    parser.add_argument('--days', type=int, default=30, help='Number of days to show')

    args = parser.parse_args()

    create_dashboard(db_path=args.db, days=args.days)


if __name__ == "__main__":
    main()
