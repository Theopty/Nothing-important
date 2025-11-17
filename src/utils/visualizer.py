"""
Visualization tools for analysis results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """Create visualizations for analysis results"""

    def __init__(self, output_dir: str = 'reports/charts'):
        """Initialize visualizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_sentiment_timeline(
        self,
        daily_aggregates: List[Dict],
        symbol: str,
        save: bool = True
    ):
        """
        Plot sentiment over time

        Args:
            daily_aggregates: List of daily aggregate data
            symbol: Stock symbol
            save: Whether to save the plot
        """
        df = pd.DataFrame(daily_aggregates)
        df['date'] = pd.to_datetime(df['date'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Sentiment timeline
        ax1.plot(df['date'], df['average_sentiment'], 'b-', linewidth=2, label='Avg Sentiment')
        ax1.fill_between(df['date'],
                         df['average_sentiment'] - df['sentiment_std'],
                         df['average_sentiment'] + df['sentiment_std'],
                         alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment Score')
        ax1.set_title(f'{symbol} - News Sentiment Timeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Article count
        ax2.bar(df['date'], df['article_count'], color='steelblue', alpha=0.7)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Articles')
        ax2.set_title(f'{symbol} - Daily Article Volume')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{symbol}_sentiment_timeline.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        plt.show()

    def plot_correlation_analysis(
        self,
        daily_aggregates: List[Dict],
        stock_data: pd.DataFrame,
        symbol: str,
        save: bool = True
    ):
        """
        Plot correlation between sentiment and stock price

        Args:
            daily_aggregates: Daily aggregates
            stock_data: Stock price data
            symbol: Stock symbol
            save: Whether to save plot
        """
        # Prepare data
        df_news = pd.DataFrame(daily_aggregates)
        df_news['date'] = pd.to_datetime(df_news['date'])
        df_news.set_index('date', inplace=True)

        df_stock = stock_data.copy()
        df_merged = df_news.join(df_stock, how='inner')

        if len(df_merged) < 2:
            print("Insufficient data for correlation plot")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # 1. Sentiment vs Price
        ax1 = axes[0]
        ax1_twin = ax1.twinx()

        ax1.plot(df_merged.index, df_merged['average_sentiment'],
                'b-', linewidth=2, label='Sentiment')
        ax1_twin.plot(df_merged.index, df_merged['Close'],
                     'r-', linewidth=2, label='Stock Price')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment Score', color='b')
        ax1_twin.set_ylabel('Stock Price ($)', color='r')
        ax1.set_title(f'{symbol} - Sentiment vs Stock Price')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)

        # Add correlation text
        corr = df_merged['average_sentiment'].corr(df_merged['Daily_Return'])
        ax1.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Sentiment vs Daily Returns
        ax2 = axes[1]
        ax2.scatter(df_merged['average_sentiment'], df_merged['Daily_Return'],
                   alpha=0.6, s=50)
        ax2.set_xlabel('Sentiment Score')
        ax2.set_ylabel('Daily Return (%)')
        ax2.set_title(f'{symbol} - Sentiment vs Daily Returns Scatter')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(df_merged['average_sentiment'].dropna(),
                      df_merged['Daily_Return'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_merged['average_sentiment'].min(),
                            df_merged['average_sentiment'].max(), 100)
        ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # 3. Volume analysis
        ax3 = axes[2]
        ax3_twin = ax3.twinx()

        ax3.bar(df_merged.index, df_merged['article_count'],
               color='steelblue', alpha=0.6, label='Article Count')
        ax3_twin.plot(df_merged.index, df_merged['Volume'] / 1e6,
                     'g-', linewidth=2, label='Trading Volume (M)')

        ax3.set_xlabel('Date')
        ax3.set_ylabel('Article Count', color='steelblue')
        ax3_twin.set_ylabel('Trading Volume (M)', color='g')
        ax3.set_title(f'{symbol} - News Volume vs Trading Volume')
        ax3.tick_params(axis='y', labelcolor='steelblue')
        ax3_twin.tick_params(axis='y', labelcolor='g')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{symbol}_correlation_analysis.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        plt.show()

    def plot_sentiment_distribution(
        self,
        analyzed_articles: List[Dict],
        save: bool = True
    ):
        """
        Plot distribution of sentiment across articles

        Args:
            analyzed_articles: List of analyzed articles
            save: Whether to save plot
        """
        sentiments = [a['sentiment']['polarity'] for a in analyzed_articles]
        sentiment_types = [a['sentiment']['sentiment'] for a in analyzed_articles]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram of polarity scores
        ax1.hist(sentiments, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
        ax1.set_xlabel('Sentiment Polarity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Sentiment Polarity Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Pie chart of sentiment types
        type_counts = pd.Series(sentiment_types).value_counts()
        colors = {'positive': '#90EE90', 'negative': '#FFB6C6', 'neutral': '#D3D3D3'}
        pie_colors = [colors.get(t, 'gray') for t in type_counts.index]

        ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
               colors=pie_colors, startangle=90)
        ax2.set_title('Sentiment Type Distribution')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'sentiment_distribution.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        plt.show()

    def plot_experiment_comparison(
        self,
        experiments: List[Dict],
        save: bool = True
    ):
        """
        Compare results from multiple experiments

        Args:
            experiments: List of experiment result dictionaries
            save: Whether to save plot
        """
        names = [exp['name'] for exp in experiments]
        correlations = [exp['metrics']['avg_correlation'] for exp in experiments]
        confidences = [exp['metrics']['avg_confidence'] for exp in experiments]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Average correlation comparison
        bars1 = ax1.bar(names, correlations, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Average Correlation')
        ax1.set_title('Experiment Comparison - Average Correlation')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Highlight best performer
        best_idx = correlations.index(max(correlations))
        bars1[best_idx].set_color('gold')

        # Signal distribution comparison
        signal_data = []
        for exp in experiments:
            signals = exp['metrics']['signals']
            signal_data.append([signals['BUY'], signals['SELL'], signals['HOLD']])

        x = np.arange(len(names))
        width = 0.25

        ax2.bar(x - width, [s[0] for s in signal_data], width, label='BUY', color='green', alpha=0.7)
        ax2.bar(x, [s[1] for s in signal_data], width, label='SELL', color='red', alpha=0.7)
        ax2.bar(x + width, [s[2] for s in signal_data], width, label='HOLD', color='gray', alpha=0.7)

        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Signal Count')
        ax2.set_title('Experiment Comparison - Signal Distribution')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'experiment_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")

        plt.show()

    def create_summary_report(
        self,
        results: Dict,
        analyzed_articles: List[Dict],
        output_file: str = 'summary_report.html'
    ):
        """
        Create an HTML summary report

        Args:
            results: Analysis results
            analyzed_articles: Analyzed articles
            output_file: Output filename
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>News-Stock Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
                .signal-buy {{ color: green; font-weight: bold; }}
                .signal-sell {{ color: red; font-weight: bold; }}
                .signal-hold {{ color: gray; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>News-Stock Sentiment Analysis Report</h1>
                <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>

            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="metric">
                    <strong>Total Articles:</strong> {len(analyzed_articles)}
                </div>
                <div class="metric">
                    <strong>Symbols Analyzed:</strong> {len(results)}
                </div>
            </div>

            <div class="section">
                <h2>Analysis Results by Symbol</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Signal</th>
                        <th>Confidence</th>
                        <th>Correlation</th>
                        <th>Articles</th>
                        <th>Avg Sentiment</th>
                    </tr>
        """

        for symbol, result in results.items():
            signal = result['signal']['signal']
            signal_class = f"signal-{signal.lower()}"
            confidence = result['signal']['confidence']

            corr = result['correlation']
            if 'error' not in corr:
                corr_value = corr['correlations']['predictive']
                corr_str = f"{corr_value:+.3f}"
            else:
                corr_str = "N/A"

            latest = result['latest_aggregate']

            html += f"""
                    <tr>
                        <td><strong>{symbol}</strong></td>
                        <td class="{signal_class}">{signal}</td>
                        <td>{confidence:.1%}</td>
                        <td>{corr_str}</td>
                        <td>{latest['article_count']}</td>
                        <td>{latest['average_sentiment']:+.3f}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        report_path = self.output_dir.parent / output_file
        with open(report_path, 'w') as f:
            f.write(html)

        print(f"Summary report saved: {report_path}")


if __name__ == "__main__":
    # Test visualizer
    viz = Visualizer()
    print("Visualizer initialized")
