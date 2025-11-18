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

    def __init__(self, api_key: Optional[str] = None, source: str = 'newsdata'):
        """
        Initialize news collector

        Args:
            api_key: API key (if not provided, will check env vars)
            source: 'newsdata' or 'worldnews'
        """
        self.source = source

        if source == 'newsdata':
            self.api_key = api_key or os.getenv('NEWSDATA_API_KEY')
            self.base_url = "https://newsdata.io/api/1/news"
        elif source == 'worldnews':
            self.api_key = api_key or os.getenv('WORLDNEWS_API_KEY')
            self.base_url = "https://api.worldnewsapi.com/search-news"
        else:
            raise ValueError(f"Unknown source: {source}. Use 'newsdata' or 'worldnews'")

        if not self.api_key:
            logger.warning(f"{source} API key not found. Using sample data for testing")

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

        if self.source == 'newsdata':
            return self._fetch_from_newsdata(start_date, end_date, categories, language, max_results)
        elif self.source == 'worldnews':
            return self._fetch_from_worldnews(start_date, end_date, categories, language, max_results)
        else:
            return self._generate_sample_news(start_date, end_date)

    def _fetch_from_newsdata(
        self,
        start_date: datetime,
        end_date: datetime,
        categories: List[str],
        language: str,
        max_results: int
    ) -> List[Dict]:
        """Fetch news from newsdata.io"""
        articles = []

        # NewsData.io parameters
        params = {
            'apikey': self.api_key,
            'language': language,
            'size': min(max_results, 50)  # newsdata.io max is 50 per request
        }

        # Add categories if specified
        if categories:
            # Map common categories to newsdata.io categories
            category_map = {
                'business': 'business',
                'technology': 'technology',
                'tech': 'technology',
                'politics': 'politics',
                'economy': 'business',
                'finance': 'business',
                'healthcare': 'health',
                'health': 'health'
            }
            mapped_categories = [category_map.get(c.lower(), c) for c in categories]
            params['category'] = ','.join(set(mapped_categories))

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'success' and 'results' in data:
                for article in data['results']:
                    # Filter by date range (newsdata.io doesn't have date params in free tier)
                    pub_date_str = article.get('pubDate')
                    if pub_date_str:
                        try:
                            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            if not (start_date <= pub_date <= end_date):
                                continue
                        except:
                            pass

                    articles.append({
                        'id': article.get('article_id'),
                        'title': article.get('title'),
                        'text': article.get('content', '') or article.get('description', ''),
                        'summary': article.get('description', ''),
                        'url': article.get('link'),
                        'image': article.get('image_url'),
                        'publish_date': article.get('pubDate'),
                        'author': ', '.join(article.get('creator', [])) if article.get('creator') else None,
                        'source': article.get('source_id', 'unknown'),
                        'category': ','.join(article.get('category', [])) if article.get('category') else 'general',
                        'fetched_at': datetime.utcnow().isoformat()
                    })

            logger.info(f"Fetched {len(articles)} articles from newsdata.io")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from newsdata.io: {e}")
            # Return sample data if API fails
            return self._generate_sample_news(start_date, end_date)

        return articles

    def _fetch_from_worldnews(
        self,
        start_date: datetime,
        end_date: datetime,
        categories: List[str],
        language: str,
        max_results: int
    ) -> List[Dict]:
        """Fetch news from WorldNewsAPI"""
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
            logger.error(f"Error fetching news from WorldNewsAPI: {e}")
            # Return sample data if API fails
            return self._generate_sample_news(start_date, end_date)

        return articles

    def fetch_news_by_query(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Fetch news articles matching a specific query (e.g., "Trump", "Oil", "Earthquake")

        Args:
            query: Search query
            start_date: Start date
            end_date: End date
            max_results: Maximum results

        Returns:
            List of articles
        """
        if self.source == 'newsdata':
            return self._fetch_query_newsdata(query, start_date, end_date, max_results)
        elif self.source == 'worldnews':
            return self._fetch_query_worldnews(query, start_date, end_date, max_results)
        else:
            return []

    def _fetch_query_newsdata(self, query: str, start_date: datetime, end_date: datetime, max_results: int) -> List[Dict]:
        """Fetch by query from NewsData.io"""
        if not self.api_key:
            logger.error("Cannot fetch news without API key")
            return []

        params = {
            'apikey': self.api_key,
            'q': query,  # Search query
            'language': 'en',
            'size': min(max_results, 50)
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            articles = []
            if data.get('status') == 'success' and 'results' in data:
                for article in data['results']:
                    pub_date_str = article.get('pubDate')
                    if pub_date_str:
                        try:
                            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            if not (start_date <= pub_date <= end_date):
                                continue
                        except:
                            pass

                    articles.append({
                        'id': article.get('article_id'),
                        'title': article.get('title'),
                        'text': article.get('content', '') or article.get('description', ''),
                        'summary': article.get('description', ''),
                        'url': article.get('link'),
                        'image': article.get('image_url'),
                        'publish_date': article.get('pubDate'),
                        'author': ', '.join(article.get('creator', [])) if article.get('creator') else None,
                        'source': article.get('source_id', 'unknown'),
                        'category': ','.join(article.get('category', [])) if article.get('category') else 'general',
                        'fetched_at': datetime.utcnow().isoformat(),
                        'query': query
                    })

            logger.info(f"Fetched {len(articles)} articles for query: {query}")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news for query '{query}': {e}")
            return []

    def _fetch_query_worldnews(self, query: str, start_date: datetime, end_date: datetime, max_results: int) -> List[Dict]:
        """Fetch by query from WorldNewsAPI"""
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
            logger.error(f"Error fetching news for query '{query}': {e}")
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
