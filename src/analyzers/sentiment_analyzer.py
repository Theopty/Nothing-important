"""
Sentiment Analyzer - Analyzes sentiment of news articles
Supports multiple backends: FinBERT, TextBlob, and OpenAI
"""

import os
import logging
from typing import Dict, List, Optional
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of text using various methods"""

    def __init__(self, method: str = 'textblob'):
        """
        Initialize sentiment analyzer

        Args:
            method: Analysis method ('textblob', 'finbert', 'openai')
        """
        self.method = method
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.openai_client = None

        if method == 'finbert':
            self._load_finbert()
        elif method == 'openai':
            self._load_openai()

    def _load_finbert(self):
        """Load FinBERT model for financial sentiment analysis"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "ProsusAI/finbert"
            logger.info(f"Loading FinBERT model: {model_name}")

            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

            logger.info("FinBERT model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading FinBERT: {e}")
            logger.info("Falling back to TextBlob")
            self.method = 'textblob'

    def _load_openai(self):
        """Load OpenAI client"""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')

            if not api_key:
                logger.warning("OpenAI API key not found, falling back to TextBlob")
                self.method = 'textblob'
                return

            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")

        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            self.method = 'textblob'

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment results
        """
        if self.method == 'finbert':
            return self._analyze_finbert(text)
        elif self.method == 'openai':
            return self._analyze_openai(text)
        else:
            return self._analyze_textblob(text)

    def _analyze_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob

        Returns:
            {
                'polarity': float (-1 to 1),
                'subjectivity': float (0 to 1),
                'sentiment': str ('positive', 'negative', 'neutral'),
                'confidence': float (0 to 1)
            }
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            # Confidence is based on polarity magnitude
            confidence = abs(polarity)

            return {
                'method': 'textblob',
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'confidence': confidence,
                'score': polarity  # Normalized score
            }

        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return self._default_sentiment()

    def _analyze_finbert(self, text: str) -> Dict:
        """
        Analyze sentiment using FinBERT (specialized for financial text)

        Returns sentiment dictionary
        """
        try:
            import torch

            # Tokenize
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Get predictions
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [positive, negative, neutral]
            scores = predictions[0].tolist()
            labels = ['positive', 'negative', 'neutral']

            # Get dominant sentiment
            max_idx = scores.index(max(scores))
            sentiment = labels[max_idx]
            confidence = scores[max_idx]

            # Calculate polarity score (-1 to 1)
            # positive - negative
            polarity = scores[0] - scores[1]

            return {
                'method': 'finbert',
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': {
                    'positive': scores[0],
                    'negative': scores[1],
                    'neutral': scores[2]
                },
                'polarity': polarity,
                'score': polarity
            }

        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return self._analyze_textblob(text)

    def _analyze_openai(self, text: str) -> Dict:
        """
        Analyze sentiment using OpenAI GPT

        Returns sentiment dictionary
        """
        try:
            prompt = f"""Analyze the sentiment of the following news article.
            Provide:
            1. Overall sentiment (positive, negative, or neutral)
            2. Confidence score (0-1)
            3. Brief reasoning

            Article: {text[:1000]}

            Respond in JSON format:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence": 0.85,
                "reasoning": "brief explanation"
            }}"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            import json
            result = json.loads(response.choices[0].message.content)

            # Convert sentiment to polarity
            polarity_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            polarity = polarity_map.get(result['sentiment'], 0.0) * result.get('confidence', 0.5)

            return {
                'method': 'openai',
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'reasoning': result.get('reasoning', ''),
                'polarity': polarity,
                'score': polarity
            }

        except Exception as e:
            logger.error(f"Error in OpenAI analysis: {e}")
            return self._analyze_textblob(text)

    def _default_sentiment(self) -> Dict:
        """Return default neutral sentiment"""
        return {
            'method': 'default',
            'sentiment': 'neutral',
            'confidence': 0.0,
            'polarity': 0.0,
            'score': 0.0
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment dictionaries
        """
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)

        return results

    def get_aggregate_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Get aggregate sentiment from multiple articles

        Args:
            articles: List of article dictionaries with 'text' or 'title'

        Returns:
            Aggregate sentiment statistics
        """
        sentiments = []

        for article in articles:
            text = article.get('text', '') or article.get('title', '')
            if text:
                sentiment = self.analyze(text)
                sentiments.append(sentiment)

        if not sentiments:
            return self._default_sentiment()

        # Calculate aggregates
        avg_polarity = sum(s['polarity'] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)

        # Count sentiments
        positive = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        neutral = sum(1 for s in sentiments if s['sentiment'] == 'neutral')

        # Determine overall sentiment
        if positive > negative and positive > neutral:
            overall = 'positive'
        elif negative > positive and negative > neutral:
            overall = 'negative'
        else:
            overall = 'neutral'

        return {
            'overall_sentiment': overall,
            'average_polarity': avg_polarity,
            'average_confidence': avg_confidence,
            'distribution': {
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            },
            'total_articles': len(sentiments)
        }


if __name__ == "__main__":
    # Test the analyzer
    analyzer = SentimentAnalyzer(method='textblob')

    test_texts = [
        "The company reported record profits, exceeding all expectations!",
        "Stock prices plummeted after disappointing earnings report.",
        "The market remained stable today with mixed signals."
    ]

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Polarity: {result['polarity']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
