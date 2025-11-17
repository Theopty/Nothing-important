"""
Database module for storing news, analysis results, and correlations
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsStockDatabase:
    """SQLite database for storing news and analysis data"""

    def __init__(self, db_path: str = 'data/news_stock_analysis.db'):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Create directory if it doesn't exist
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")

    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()

        # News articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT,
                text TEXT,
                summary TEXT,
                url TEXT,
                publish_date TEXT,
                author TEXT,
                source TEXT,
                category TEXT,
                fetched_at TEXT,
                sentiment_polarity REAL,
                sentiment_type TEXT,
                sentiment_confidence REAL,
                entities_json TEXT,
                industries_json TEXT,
                sentiment_score REAL,
                entity_score REAL,
                impact_score REAL,
                total_score REAL,
                analyzed_at TEXT
            )
        """)

        # Stock data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                daily_return REAL,
                volume_change REAL,
                fetched_at TEXT,
                UNIQUE(symbol, date)
            )
        """)

        # Daily aggregates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_aggregates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                symbol TEXT,
                article_count INTEGER,
                average_sentiment REAL,
                sentiment_std REAL,
                average_score REAL,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                source_count INTEGER,
                industries_json TEXT,
                companies_json TEXT,
                created_at TEXT,
                UNIQUE(date, symbol)
            )
        """)

        # Correlation results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                period_start TEXT,
                period_end TEXT,
                days_analyzed INTEGER,
                sentiment_price_corr REAL,
                sentiment_volume_corr REAL,
                score_price_corr REAL,
                predictive_corr REAL,
                significant_days_count INTEGER,
                created_at TEXT
            )
        """)

        # Experiments table for tracking different algorithm versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                version TEXT,
                config_json TEXT,
                results_json TEXT,
                metrics_json TEXT,
                created_at TEXT
            )
        """)

        self.conn.commit()
        logger.info("Database tables created/verified")

    def save_article(self, article: Dict):
        """Save an analyzed article to database"""
        try:
            cursor = self.conn.cursor()

            sentiment = article.get('sentiment', {})
            entities = article.get('entities', {})
            scores = article.get('scores', {})

            cursor.execute("""
                INSERT OR REPLACE INTO articles (
                    id, title, text, summary, url, publish_date, author, source, category,
                    fetched_at, sentiment_polarity, sentiment_type, sentiment_confidence,
                    entities_json, industries_json, sentiment_score, entity_score,
                    impact_score, total_score, analyzed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.get('id'),
                article.get('title'),
                article.get('text'),
                article.get('summary'),
                article.get('url'),
                article.get('publish_date'),
                article.get('author'),
                article.get('source'),
                article.get('category'),
                article.get('fetched_at'),
                sentiment.get('polarity'),
                sentiment.get('sentiment'),
                sentiment.get('confidence'),
                json.dumps(entities),
                json.dumps(article.get('industries', [])),
                scores.get('sentiment_score'),
                scores.get('entity_score'),
                scores.get('impact_score'),
                scores.get('total_score'),
                article.get('analyzed_at')
            ))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Error saving article: {e}")

    def save_articles_batch(self, articles: List[Dict]):
        """Save multiple articles"""
        for article in articles:
            self.save_article(article)

    def save_stock_data(self, symbol: str, stock_df: pd.DataFrame):
        """Save stock data to database"""
        try:
            for idx, row in stock_df.iterrows():
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO stock_data (
                        symbol, date, open, high, low, close, volume,
                        daily_return, volume_change, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    idx.isoformat(),
                    row.get('Open'),
                    row.get('High'),
                    row.get('Low'),
                    row.get('Close'),
                    int(row.get('Volume', 0)),
                    row.get('Daily_Return'),
                    row.get('Volume_Change'),
                    datetime.utcnow().isoformat()
                ))

            self.conn.commit()
            logger.info(f"Saved {len(stock_df)} stock data points for {symbol}")

        except Exception as e:
            logger.error(f"Error saving stock data: {e}")

    def save_daily_aggregate(self, aggregate: Dict):
        """Save daily aggregate statistics"""
        try:
            cursor = self.conn.cursor()

            dist = aggregate.get('sentiment_distribution', {})

            cursor.execute("""
                INSERT OR REPLACE INTO daily_aggregates (
                    date, symbol, article_count, average_sentiment, sentiment_std,
                    average_score, positive_count, negative_count, neutral_count,
                    source_count, industries_json, companies_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                aggregate.get('date'),
                aggregate.get('symbol', 'GENERAL'),
                aggregate.get('article_count'),
                aggregate.get('average_sentiment'),
                aggregate.get('sentiment_std'),
                aggregate.get('average_score'),
                dist.get('positive'),
                dist.get('negative'),
                dist.get('neutral'),
                aggregate.get('source_count'),
                json.dumps(aggregate.get('industries', [])),
                json.dumps(aggregate.get('companies_mentioned', [])),
                datetime.utcnow().isoformat()
            ))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Error saving daily aggregate: {e}")

    def save_correlation(self, correlation: Dict):
        """Save correlation analysis results"""
        try:
            cursor = self.conn.cursor()

            period = correlation.get('period', {})
            corrs = correlation.get('correlations', {})

            cursor.execute("""
                INSERT INTO correlations (
                    symbol, period_start, period_end, days_analyzed,
                    sentiment_price_corr, sentiment_volume_corr, score_price_corr,
                    predictive_corr, significant_days_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                correlation.get('symbol'),
                period.get('start'),
                period.get('end'),
                period.get('days'),
                corrs.get('sentiment_price'),
                corrs.get('sentiment_volume'),
                corrs.get('score_price'),
                corrs.get('predictive'),
                correlation.get('significant_days_count'),
                datetime.utcnow().isoformat()
            ))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Error saving correlation: {e}")

    def save_experiment(self, name: str, version: str, config: Dict, results: Dict, metrics: Dict):
        """Save experiment results for tracking algorithm performance"""
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                INSERT INTO experiments (
                    experiment_name, version, config_json, results_json, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name,
                version,
                json.dumps(config),
                json.dumps(results),
                json.dumps(metrics),
                datetime.utcnow().isoformat()
            ))

            self.conn.commit()
            logger.info(f"Saved experiment: {name}")

        except Exception as e:
            logger.error(f"Error saving experiment: {e}")

    def get_articles_by_date(self, start_date: str, end_date: str) -> List[Dict]:
        """Retrieve articles for a date range"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM articles
            WHERE publish_date BETWEEN ? AND ?
            ORDER BY publish_date DESC
        """, (start_date, end_date))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve stock data as DataFrame"""
        query = """
            SELECT * FROM stock_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
        """

        df = pd.read_sql_query(query, self.conn, params=(symbol, start_date, end_date))

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        return df

    def get_experiments(self, limit: int = 10) -> List[Dict]:
        """Get recent experiments"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM experiments
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        experiments = []

        for row in rows:
            exp = dict(row)
            exp['config'] = json.loads(exp['config_json'])
            exp['results'] = json.loads(exp['results_json'])
            exp['metrics'] = json.loads(exp['metrics_json'])
            experiments.append(exp)

        return experiments

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    # Test database
    db = NewsStockDatabase('data/test.db')
    print("Database initialized successfully")
    db.close()
