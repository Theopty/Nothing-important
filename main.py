#!/usr/bin/env python3
"""
News-Stock Sentiment Algorithm - Main Runner
This script orchestrates the entire analysis pipeline
"""

import argparse
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.collectors.news_collector import NewsCollector
from src.collectors.stock_collector import StockCollector
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.entity_analyzer import EntityAnalyzer
from src.algorithm.news_stock_algorithm import NewsStockAlgorithm
from src.storage.database import NewsStockDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsStockAnalyzer:
    """Main orchestrator for news-stock analysis"""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize all components"""
        logger.info("Initializing News-Stock Analyzer...")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        news_source = self.config.get('news', {}).get('source', 'newsdata')
        self.news_collector = NewsCollector(source=news_source)
        self.stock_collector = StockCollector()
        self.sentiment_analyzer = SentimentAnalyzer(
            method=self.config['ai_analysis']['sentiment']['model']
        )
        self.entity_analyzer = EntityAnalyzer()
        self.algorithm = NewsStockAlgorithm(config_path)
        self.database = NewsStockDatabase()

        logger.info("All components initialized successfully")

    def run_analysis(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        symbols: list = None,
        save_to_db: bool = True
    ):
        """
        Run complete analysis pipeline

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            symbols: List of stock symbols to analyze
            save_to_db: Whether to save results to database
        """
        # Default date range
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=self.config['news']['lookback_days'])

        # Default symbols
        if not symbols:
            symbols = self.config['stocks']['watchlist']

        logger.info(f"Running analysis from {start_date.date()} to {end_date.date()}")
        logger.info(f"Analyzing {len(symbols)} symbols: {symbols}")

        # Step 1: Collect news
        logger.info("Step 1: Collecting news articles...")
        articles = self.news_collector.fetch_news(
            start_date=start_date,
            end_date=end_date,
            categories=self.config['news']['categories'],
            max_results=self.config['news']['max_articles_per_day']
        )
        logger.info(f"Collected {len(articles)} articles")

        # Step 2: Analyze news
        logger.info("Step 2: Analyzing news articles...")
        analyzed_articles = self.algorithm.analyze_news_batch(
            articles,
            self.sentiment_analyzer,
            self.entity_analyzer
        )
        logger.info(f"Analyzed {len(analyzed_articles)} articles")

        # Save articles to database
        if save_to_db:
            logger.info("Saving articles to database...")
            self.database.save_articles_batch(analyzed_articles)

        # Step 3: Collect and analyze stock data for each symbol
        results = {}

        for symbol in symbols:
            logger.info(f"\nAnalyzing {symbol}...")

            try:
                # Collect stock data
                stock_data = self.stock_collector.fetch_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )

                if stock_data is None or stock_data.empty:
                    logger.warning(f"No stock data for {symbol}, skipping")
                    continue

                # Save stock data
                if save_to_db:
                    self.database.save_stock_data(symbol, stock_data)

                # Generate daily aggregates
                logger.info(f"Generating daily aggregates for {symbol}...")
                daily_aggregates = []

                current_date = start_date
                while current_date <= end_date:
                    daily_agg = self.algorithm.aggregate_daily_sentiment(
                        analyzed_articles,
                        current_date
                    )
                    daily_agg['symbol'] = symbol
                    daily_aggregates.append(daily_agg)

                    if save_to_db:
                        self.database.save_daily_aggregate(daily_agg)

                    current_date += timedelta(days=1)

                # Correlate news with stock movements
                logger.info(f"Calculating correlations for {symbol}...")
                correlation = self.algorithm.correlate_with_stock(
                    daily_aggregates,
                    stock_data,
                    symbol
                )

                if save_to_db and 'error' not in correlation:
                    self.database.save_correlation(correlation)

                # Generate trading signals
                latest_aggregate = daily_aggregates[-1]
                signal = self.algorithm.generate_trading_signals(
                    correlation,
                    latest_aggregate
                )

                results[symbol] = {
                    'stock_data': stock_data,
                    'daily_aggregates': daily_aggregates,
                    'correlation': correlation,
                    'signal': signal,
                    'latest_aggregate': latest_aggregate
                }

                # Print summary
                self._print_symbol_summary(symbol, results[symbol])

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        # Print overall summary
        self._print_overall_summary(results, analyzed_articles)

        return results

    def _print_symbol_summary(self, symbol: str, result: Dict):
        """Print summary for a single symbol"""
        print(f"\n{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"{'='*60}")

        corr = result['correlation']
        if 'error' not in corr:
            print(f"\nCorrelations:")
            print(f"  Sentiment vs Price:   {corr['correlations']['sentiment_price']:+.3f}")
            print(f"  Sentiment vs Volume:  {corr['correlations']['sentiment_volume']:+.3f}")
            print(f"  Score vs Price:       {corr['correlations']['score_price']:+.3f}")
            print(f"  Predictive (next day): {corr['correlations']['predictive']:+.3f}")

        signal = result['signal']
        print(f"\nTrading Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']:.2%}")
        if signal['reasoning']:
            print(f"Reasoning:")
            for reason in signal['reasoning']:
                print(f"  - {reason}")

        latest = result['latest_aggregate']
        print(f"\nLatest Sentiment:")
        print(f"  Articles: {latest['article_count']}")
        print(f"  Avg Sentiment: {latest['average_sentiment']:+.3f}")
        print(f"  Distribution: +{latest['sentiment_distribution']['positive']} "
              f"-{latest['sentiment_distribution']['negative']} "
              f"={latest['sentiment_distribution']['neutral']}")

    def _print_overall_summary(self, results: Dict, articles: List):
        """Print overall analysis summary"""
        print(f"\n{'='*60}")
        print(f"OVERALL ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total articles analyzed: {len(articles)}")
        print(f"Symbols analyzed: {len(results)}")

        # Count signals
        buy_signals = sum(1 for r in results.values() if r['signal']['signal'] == 'BUY')
        sell_signals = sum(1 for r in results.values() if r['signal']['signal'] == 'SELL')
        hold_signals = sum(1 for r in results.values() if r['signal']['signal'] == 'HOLD')

        print(f"\nSignal Distribution:")
        print(f"  BUY:  {buy_signals}")
        print(f"  SELL: {sell_signals}")
        print(f"  HOLD: {hold_signals}")

        # Average correlations
        valid_corrs = [
            r['correlation']['correlations']['predictive']
            for r in results.values()
            if 'error' not in r['correlation']
        ]

        if valid_corrs:
            avg_predictive_corr = sum(valid_corrs) / len(valid_corrs)
            print(f"\nAverage Predictive Correlation: {avg_predictive_corr:+.3f}")

    def run_experiment(self, experiment_name: str, custom_config: Dict = None):
        """
        Run an experiment with custom configuration

        Args:
            experiment_name: Name for this experiment
            custom_config: Custom configuration overrides
        """
        logger.info(f"Running experiment: {experiment_name}")

        # Merge custom config if provided
        if custom_config:
            # Update algorithm weights
            if 'weights' in custom_config:
                self.algorithm.weights.update(custom_config['weights'])
            if 'scoring' in custom_config:
                self.algorithm.scoring.update(custom_config['scoring'])

        # Run analysis
        results = self.run_analysis()

        # Calculate metrics
        metrics = self._calculate_experiment_metrics(results)

        # Save experiment
        self.database.save_experiment(
            name=experiment_name,
            version=self.algorithm.version,
            config={'weights': self.algorithm.weights, 'scoring': self.algorithm.scoring},
            results={k: {'signal': v['signal']} for k, v in results.items()},
            metrics=metrics
        )

        logger.info(f"Experiment '{experiment_name}' completed and saved")

        return results, metrics

    def _calculate_experiment_metrics(self, results: Dict) -> Dict:
        """Calculate metrics for experiment tracking"""
        metrics = {
            'total_symbols': len(results),
            'avg_correlation': 0.0,
            'signals': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'avg_confidence': 0.0
        }

        correlations = []
        confidences = []

        for result in results.values():
            # Signal counts
            signal = result['signal']['signal']
            metrics['signals'][signal] += 1

            # Correlations
            if 'error' not in result['correlation']:
                corr = result['correlation']['correlations']['predictive']
                correlations.append(abs(corr))

            # Confidences
            confidences.append(result['signal']['confidence'])

        if correlations:
            metrics['avg_correlation'] = sum(correlations) / len(correlations)
        if confidences:
            metrics['avg_confidence'] = sum(confidences) / len(confidences)

        return metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='News-Stock Sentiment Analysis Algorithm')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to analyze')
    parser.add_argument('--experiment', help='Run as experiment with this name')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save to database')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = NewsStockAnalyzer(config_path=args.config)

    # Set date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    # Run analysis or experiment
    if args.experiment:
        results, metrics = analyzer.run_experiment(args.experiment)
        print(f"\nExperiment Metrics:")
        print(f"  Avg Correlation: {metrics['avg_correlation']:.3f}")
        print(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
    else:
        results = analyzer.run_analysis(
            start_date=start_date,
            end_date=end_date,
            symbols=args.symbols,
            save_to_db=not args.no_save
        )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
