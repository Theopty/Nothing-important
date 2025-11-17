"""
News Collector - Fetches news articles from WorldNewsAPI and other sources
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsCollector:
    """Collects news articles from various sources"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('WORLDNEWS_API_KEY')
        self.base_url = "https://api.worldnewsapi.com/search-news"

        if not self.api_key:
            logger.warning("WorldNewsAPI key not found. Set WORLDNEWS_API_KEY in .env file")

    def fetch_news(
        self,
        start_date: datetime,
        end_date: datetime,
        categories: List[str] = None,
        language: str = 'en',
        max_results: int = 100
    ) -> List[Dict]:
        """
        Fetch news articles for a date range

        Args:
            start_date: Start date for news search
            end_date: End date for news search
            categories: List of categories (business, technology, etc.)
            language: Language code (default: en)
            max_results: Maximum number of articles to fetch

        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            logger.error("Cannot fetch news without API key")
            return self._generate_sample_news(start_date, end_date)

        articles = []

        # WorldNewsAPI parameters
        params = {
            'api-key': self.api_key,
            'earliest-publish-date': start_date.strftime('%Y-%m-%d'),
            'latest-publish-date': end_date.strftime('%Y-%m-%d'),
            'language': language,
            'number': max_results,
            'sort': 'publish-time',
            'sort-direction': 'DESC'
        }

        # Add categories if specified
        if categories:
            # WorldNewsAPI uses text search, so we'll search for category terms
            category_terms = ' OR '.join(categories)
            params['text'] = category_terms

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'news' in data:
                for article in data['news']:
                    articles.append({
                        'id': article.get('id'),
                        'title': article.get('title'),
                        'text': article.get('text', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('url'),
                        'image': article.get('image'),
                        'publish_date': article.get('publish_date'),
                        'author': article.get('author'),
                        'source': article.get('source_country', 'unknown'),
                        'category': self._infer_category(article.get('title', '') + ' ' + article.get('text', '')),
                        'fetched_at': datetime.utcnow().isoformat()
                    })

            logger.info(f"Fetched {len(articles)} articles from WorldNewsAPI")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news: {e}")
            # Return sample data if API fails
            return self._generate_sample_news(start_date, end_date)

        return articles

    def fetch_news_by_query(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Fetch news articles matching a specific query (e.g., company name, topic)

        Args:
            query: Search query
            start_date: Start date
            end_date: End date
            max_results: Maximum results

        Returns:
            List of articles
        """
        if not self.api_key:
            logger.error("Cannot fetch news without API key")
            return []

        params = {
            'api-key': self.api_key,
            'text': query,
            'earliest-publish-date': start_date.strftime('%Y-%m-%d'),
            'latest-publish-date': end_date.strftime('%Y-%m-%d'),
            'number': max_results,
            'sort': 'publish-time',
            'sort-direction': 'DESC'
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            articles = []

            if 'news' in data:
                for article in data['news']:
                    articles.append({
                        'id': article.get('id'),
                        'title': article.get('title'),
                        'text': article.get('text', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('url'),
                        'publish_date': article.get('publish_date'),
                        'source': article.get('source_country', 'unknown'),
                        'query': query,
                        'fetched_at': datetime.utcnow().isoformat()
                    })

            logger.info(f"Fetched {len(articles)} articles for query: {query}")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news by query: {e}")
            return []

    def _infer_category(self, text: str) -> str:
        """Infer article category from text"""
        text_lower = text.lower()

        business_keywords = ['market', 'stock', 'trading', 'finance', 'investor', 'economy', 'bank']
        tech_keywords = ['technology', 'software', 'ai', 'tech', 'digital', 'cyber', 'data']
        health_keywords = ['health', 'medical', 'drug', 'pharma', 'hospital', 'disease']
        energy_keywords = ['energy', 'oil', 'gas', 'renewable', 'power', 'electric']

        for keyword in business_keywords:
            if keyword in text_lower:
                return 'business'

        for keyword in tech_keywords:
            if keyword in text_lower:
                return 'technology'

        for keyword in health_keywords:
            if keyword in text_lower:
                return 'healthcare'

        for keyword in energy_keywords:
            if keyword in text_lower:
                return 'energy'

        return 'general'

    def _generate_sample_news(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate sample news data for testing when API is not available"""
        logger.info("Generating sample news data for testing...")

        sample_articles = [
            {
                'id': 'sample_1',
                'title': 'Tech Giant Announces Record Quarterly Earnings',
                'text': 'Major technology company reported exceptional quarterly results, beating analyst expectations by 15%. The strong performance was driven by cloud services and AI products.',
                'summary': 'Tech company beats earnings expectations with strong cloud and AI growth.',
                'url': 'https://example.com/news/1',
                'publish_date': (end_date - timedelta(hours=2)).isoformat(),
                'author': 'Jane Reporter',
                'source': 'Financial Times',
                'category': 'technology',
                'fetched_at': datetime.utcnow().isoformat()
            },
            {
                'id': 'sample_2',
                'title': 'Oil Prices Surge Amid Supply Concerns',
                'text': 'Crude oil prices jumped 5% today as geopolitical tensions raised concerns about global supply chains. Analysts predict continued volatility in energy markets.',
                'summary': 'Oil prices rise due to supply concerns and geopolitical tensions.',
                'url': 'https://example.com/news/2',
                'publish_date': (end_date - timedelta(hours=5)).isoformat(),
                'author': 'John Smith',
                'source': 'Reuters',
                'category': 'energy',
                'fetched_at': datetime.utcnow().isoformat()
            },
            {
                'id': 'sample_3',
                'title': 'Federal Reserve Signals Potential Rate Cuts',
                'text': 'The Federal Reserve indicated it may consider interest rate reductions in the coming months, citing cooling inflation. Markets rallied on the dovish comments.',
                'summary': 'Fed hints at possible rate cuts, markets respond positively.',
                'url': 'https://example.com/news/3',
                'publish_date': (end_date - timedelta(hours=8)).isoformat(),
                'author': 'Sarah Williams',
                'source': 'Bloomberg',
                'category': 'business',
                'fetched_at': datetime.utcnow().isoformat()
            },
            {
                'id': 'sample_4',
                'title': 'Pharmaceutical Company Faces FDA Setback',
                'text': 'Leading drug manufacturer received unexpected rejection from FDA for its new treatment, causing shares to plummet 12% in after-hours trading.',
                'summary': 'Pharma company stock drops after FDA rejects new drug application.',
                'url': 'https://example.com/news/4',
                'publish_date': (end_date - timedelta(hours=12)).isoformat(),
                'author': 'Michael Chen',
                'source': 'Wall Street Journal',
                'category': 'healthcare',
                'fetched_at': datetime.utcnow().isoformat()
            },
            {
                'id': 'sample_5',
                'title': 'Electric Vehicle Sales Break Records',
                'text': 'Electric vehicle sales reached all-time highs this quarter, with multiple manufacturers reporting strong demand. Industry analysts are optimistic about continued growth.',
                'summary': 'EV sales hit record levels across multiple manufacturers.',
                'url': 'https://example.com/news/5',
                'publish_date': (end_date - timedelta(hours=18)).isoformat(),
                'author': 'Lisa Anderson',
                'source': 'CNBC',
                'category': 'technology',
                'fetched_at': datetime.utcnow().isoformat()
            }
        ]

        return sample_articles


if __name__ == "__main__":
    # Test the collector
    collector = NewsCollector()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)

    articles = collector.fetch_news(start_date, end_date)
    print(f"Collected {len(articles)} articles")

    if articles:
        print(f"\nSample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"Category: {articles[0]['category']}")
        print(f"Date: {articles[0]['publish_date']}")
