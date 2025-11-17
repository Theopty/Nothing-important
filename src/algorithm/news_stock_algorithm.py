"""
News-Stock Correlation Algorithm
Main algorithm that combines news analysis with stock data to generate signals
"""

import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsStockAlgorithm:
    """
    Main algorithm for correlating news sentiment with stock movements
    Highly configurable through config.yaml
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize algorithm with configuration

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.version = self.config['algorithm']['version']
        self.weights = self.config['algorithm']['weights']
        self.scoring = self.config['algorithm']['scoring']

        logger.info(f"Initialized NewsStockAlgorithm v{self.version}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'algorithm': {
                'version': '1.0',
                'weights': {
                    'sentiment_score': 0.35,
                    'entity_relevance': 0.25,
                    'source_diversity': 0.15,
                    'volume_spike': 0.15,
                    'framing_consistency': 0.10
                },
                'scoring': {
                    'positive_sentiment_multiplier': 1.2,
                    'negative_sentiment_multiplier': 1.5,
                    'neutral_sentiment_multiplier': 0.5,
                    'direct_company_mention': 2.0,
                    'industry_mention': 1.0,
                    'multiple_sources_bonus': 1.3,
                    'volume_spike_threshold': 1.5,
                    'volume_spike_multiplier': 1.4
                }
            }
        }

    def analyze_news_batch(
        self,
        articles: List[Dict],
        sentiment_analyzer,
        entity_analyzer,
        target_symbol: str = None
    ) -> List[Dict]:
        """
        Analyze a batch of news articles

        Args:
            articles: List of article dictionaries
            sentiment_analyzer: Sentiment analyzer instance
            entity_analyzer: Entity analyzer instance
            target_symbol: Optional stock symbol to focus on

        Returns:
            List of analyzed articles with scores
        """
        analyzed_articles = []

        for article in articles:
            try:
                analyzed = self._analyze_single_article(
                    article,
                    sentiment_analyzer,
                    entity_analyzer,
                    target_symbol
                )
                analyzed_articles.append(analyzed)
            except Exception as e:
                logger.error(f"Error analyzing article {article.get('id')}: {e}")

        return analyzed_articles

    def _analyze_single_article(
        self,
        article: Dict,
        sentiment_analyzer,
        entity_analyzer,
        target_symbol: str = None
    ) -> Dict:
        """Analyze a single article and compute scores"""

        # Combine title and text for analysis
        text = f"{article.get('title', '')} {article.get('text', '')}"

        # Sentiment analysis
        sentiment = sentiment_analyzer.analyze(text)

        # Entity extraction
        entities = entity_analyzer.extract_entities(text)
        industries = entity_analyzer.identify_industry(text)
        entities['industries'] = industries

        # Calculate component scores
        sentiment_score = self._calculate_sentiment_score(sentiment)
        entity_score = self._calculate_entity_score(entities, target_symbol)
        impact_score = self._calculate_impact_score(article, sentiment, entities)

        # Combine scores using configured weights
        total_score = (
            sentiment_score * self.weights['sentiment_score'] +
            entity_score * self.weights['entity_relevance'] +
            impact_score * self.weights.get('impact', 0.1)
        )

        # Add analysis results to article
        article_analyzed = article.copy()
        article_analyzed.update({
            'sentiment': sentiment,
            'entities': entities,
            'industries': industries,
            'scores': {
                'sentiment_score': sentiment_score,
                'entity_score': entity_score,
                'impact_score': impact_score,
                'total_score': total_score
            },
            'analyzed_at': datetime.utcnow().isoformat()
        })

        return article_analyzed

    def _calculate_sentiment_score(self, sentiment: Dict) -> float:
        """
        Calculate sentiment component score

        Args:
            sentiment: Sentiment analysis result

        Returns:
            Normalized score (0-1)
        """
        polarity = sentiment.get('polarity', 0.0)
        confidence = sentiment.get('confidence', 0.5)
        sentiment_type = sentiment.get('sentiment', 'neutral')

        # Apply multipliers based on sentiment type
        if sentiment_type == 'positive':
            multiplier = self.scoring['positive_sentiment_multiplier']
        elif sentiment_type == 'negative':
            multiplier = self.scoring['negative_sentiment_multiplier']
        else:
            multiplier = self.scoring['neutral_sentiment_multiplier']

        # Score = abs(polarity) * confidence * multiplier
        score = abs(polarity) * confidence * multiplier

        # Normalize to 0-1
        return min(score, 1.0)

    def _calculate_entity_score(self, entities: Dict, target_symbol: str = None) -> float:
        """
        Calculate entity relevance score

        Args:
            entities: Extracted entities
            target_symbol: Target stock symbol

        Returns:
            Normalized score (0-1)
        """
        score = 0.0

        # Company mentions
        num_companies = len(entities.get('companies', []))
        if num_companies > 0:
            score += min(num_companies * 0.2, 0.5)

        # Ticker mentions
        num_tickers = len(entities.get('tickers', []))
        if num_tickers > 0:
            score += min(num_tickers * 0.15, 0.3)

        # People mentions (executives, analysts)
        num_people = len(entities.get('people', []))
        if num_people > 0:
            score += min(num_people * 0.1, 0.2)

        return min(score, 1.0)

    def _calculate_impact_score(
        self,
        article: Dict,
        sentiment: Dict,
        entities: Dict
    ) -> float:
        """
        Calculate potential market impact score

        Args:
            article: Article data
            sentiment: Sentiment analysis
            entities: Extracted entities

        Returns:
            Impact score (0-1)
        """
        score = 0.5  # Base score

        # Strong sentiment increases impact
        if abs(sentiment.get('polarity', 0)) > 0.6:
            score += 0.2

        # Multiple company mentions increase impact
        if len(entities.get('companies', [])) > 1:
            score += 0.1

        # Industry-wide news has more impact
        if len(entities.get('industries', [])) > 1:
            score += 0.15

        # Recent articles have more impact
        publish_date = article.get('publish_date')
        if publish_date:
            try:
                pub_datetime = pd.to_datetime(publish_date)
                hours_old = (datetime.utcnow() - pub_datetime).total_seconds() / 3600

                # Decay impact over time
                if hours_old < 6:
                    score += 0.1
                elif hours_old < 24:
                    score += 0.05
            except:
                pass

        return min(score, 1.0)

    def aggregate_daily_sentiment(
        self,
        analyzed_articles: List[Dict],
        date: datetime
    ) -> Dict:
        """
        Aggregate sentiment for a specific day

        Args:
            analyzed_articles: List of analyzed articles
            date: Target date

        Returns:
            Daily aggregate statistics
        """
        # Filter articles for the specific day
        day_articles = []
        for article in analyzed_articles:
            pub_date = article.get('publish_date')
            if pub_date:
                try:
                    pub_datetime = pd.to_datetime(pub_date)
                    if pub_datetime.date() == date.date():
                        day_articles.append(article)
                except:
                    continue

        if not day_articles:
            return self._empty_daily_aggregate(date)

        # Calculate aggregates
        sentiments = [a['sentiment']['polarity'] for a in day_articles]
        scores = [a['scores']['total_score'] for a in day_articles]

        # Sentiment distribution
        positive = sum(1 for a in day_articles if a['sentiment']['sentiment'] == 'positive')
        negative = sum(1 for a in day_articles if a['sentiment']['sentiment'] == 'negative')
        neutral = sum(1 for a in day_articles if a['sentiment']['sentiment'] == 'neutral')

        # Source diversity
        sources = set(a.get('source', 'unknown') for a in day_articles)

        # Industry coverage
        all_industries = []
        for article in day_articles:
            all_industries.extend(article.get('industries', []))

        # Company mentions
        all_companies = []
        for article in day_articles:
            all_companies.extend(article.get('entities', {}).get('companies', []))

        return {
            'date': date.date().isoformat(),
            'article_count': len(day_articles),
            'average_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'average_score': np.mean(scores),
            'sentiment_distribution': {
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            },
            'source_count': len(sources),
            'sources': list(sources),
            'industries': list(set(all_industries)),
            'companies_mentioned': list(set(all_companies)),
            'top_articles': sorted(
                day_articles,
                key=lambda x: x['scores']['total_score'],
                reverse=True
            )[:5]
        }

    def _empty_daily_aggregate(self, date: datetime) -> Dict:
        """Return empty aggregate for days with no articles"""
        return {
            'date': date.date().isoformat(),
            'article_count': 0,
            'average_sentiment': 0.0,
            'sentiment_std': 0.0,
            'average_score': 0.0,
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'source_count': 0,
            'sources': [],
            'industries': [],
            'companies_mentioned': [],
            'top_articles': []
        }

    def correlate_with_stock(
        self,
        daily_aggregates: List[Dict],
        stock_data: pd.DataFrame,
        symbol: str
    ) -> Dict:
        """
        Correlate news sentiment with stock movements

        Args:
            daily_aggregates: List of daily sentiment aggregates
            stock_data: Stock price DataFrame
            symbol: Stock symbol

        Returns:
            Correlation analysis results
        """
        try:
            # Create DataFrame from daily aggregates
            df_news = pd.DataFrame(daily_aggregates)
            df_news['date'] = pd.to_datetime(df_news['date'])
            df_news.set_index('date', inplace=True)

            # Prepare stock data
            df_stock = stock_data.copy()
            df_stock.index = pd.to_datetime(df_stock.index)

            # Merge news and stock data
            df_merged = df_news.join(df_stock, how='inner')

            if len(df_merged) < 2:
                logger.warning("Insufficient data for correlation")
                return {'error': 'insufficient_data'}

            # Calculate correlations
            sentiment_price_corr = df_merged['average_sentiment'].corr(df_merged['Daily_Return'])
            sentiment_volume_corr = df_merged['average_sentiment'].corr(df_merged['Volume_Change'])
            score_price_corr = df_merged['average_score'].corr(df_merged['Daily_Return'])

            # Analyze lead/lag relationships
            # Check if sentiment predicts next-day returns
            df_merged['next_day_return'] = df_merged['Daily_Return'].shift(-1)
            predictive_corr = df_merged['average_sentiment'].corr(df_merged['next_day_return'])

            # Identify significant news days (high sentiment + high volume)
            df_merged['significant_day'] = (
                (abs(df_merged['average_sentiment']) > 0.5) &
                (df_merged['article_count'] > df_merged['article_count'].mean())
            )

            significant_days = df_merged[df_merged['significant_day']]

            return {
                'symbol': symbol,
                'period': {
                    'start': df_merged.index.min().isoformat(),
                    'end': df_merged.index.max().isoformat(),
                    'days': len(df_merged)
                },
                'correlations': {
                    'sentiment_price': sentiment_price_corr,
                    'sentiment_volume': sentiment_volume_corr,
                    'score_price': score_price_corr,
                    'predictive': predictive_corr
                },
                'significant_days_count': len(significant_days),
                'significant_days': significant_days[['average_sentiment', 'article_count', 'Daily_Return']].to_dict('records'),
                'statistics': {
                    'avg_sentiment': df_merged['average_sentiment'].mean(),
                    'avg_daily_return': df_merged['Daily_Return'].mean(),
                    'sentiment_volatility': df_merged['average_sentiment'].std(),
                    'price_volatility': df_merged['Daily_Return'].std()
                }
            }

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {'error': str(e)}

    def generate_trading_signals(
        self,
        correlation_results: Dict,
        daily_aggregate: Dict,
        threshold: float = 0.6
    ) -> Dict:
        """
        Generate potential trading signals based on analysis

        Args:
            correlation_results: Results from correlation analysis
            daily_aggregate: Today's aggregate sentiment
            threshold: Signal threshold

        Returns:
            Trading signal dictionary
        """
        signal = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': correlation_results.get('symbol'),
            'signal': 'HOLD',
            'confidence': 0.0,
            'reasoning': []
        }

        # Check sentiment
        avg_sentiment = daily_aggregate.get('average_sentiment', 0)
        article_count = daily_aggregate.get('article_count', 0)
        avg_score = daily_aggregate.get('average_score', 0)

        # Signal logic
        if avg_score > threshold and article_count > 5:
            if avg_sentiment > 0.3:
                signal['signal'] = 'BUY'
                signal['reasoning'].append(f'Strong positive sentiment ({avg_sentiment:.2f})')
            elif avg_sentiment < -0.3:
                signal['signal'] = 'SELL'
                signal['reasoning'].append(f'Strong negative sentiment ({avg_sentiment:.2f})')

            # Check predictive correlation
            predictive_corr = correlation_results.get('correlations', {}).get('predictive', 0)
            if abs(predictive_corr) > 0.5:
                signal['reasoning'].append(f'Historical predictive correlation: {predictive_corr:.2f}')
                signal['confidence'] = min(abs(predictive_corr), 0.9)
            else:
                signal['confidence'] = 0.3

            # Adjust for article volume
            if article_count > 10:
                signal['reasoning'].append(f'High article volume ({article_count} articles)')
                signal['confidence'] += 0.1

        return signal


if __name__ == "__main__":
    # Test the algorithm
    algorithm = NewsStockAlgorithm('config.yaml')
    print(f"Algorithm initialized: v{algorithm.version}")
    print(f"Weights: {algorithm.weights}")
