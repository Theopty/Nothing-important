"""
Entity Analyzer - Extracts entities (people, companies, organizations) from text
"""

import re
import logging
from typing import Dict, List, Set
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityAnalyzer:
    """Extracts and analyzes entities from text"""

    def __init__(self):
        """Initialize entity analyzer"""
        self.spacy_model = None
        self._load_spacy()

        # Common stock ticker patterns
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

        # Known companies and ticker mapping
        self.company_tickers = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'nvidia': 'NVDA',
            'amd': 'AMD',
            'intel': 'INTC',
            'netflix': 'NFLX',
            'disney': 'DIS',
            'walmart': 'WMT',
            'jpmorgan': 'JPM',
            'visa': 'V',
            'mastercard': 'MA',
            'pfizer': 'PFE',
            'johnson & johnson': 'JNJ',
            'exxon': 'XOM',
            'chevron': 'CVX'
        }

    def _load_spacy(self):
        """Load spaCy model for NER"""
        try:
            import spacy
            # Try to load English model
            try:
                self.spacy_model = spacy.load('en_core_web_sm')
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                logger.info("Using basic entity extraction without spaCy")

        except ImportError:
            logger.warning("spaCy not installed. Using basic entity extraction")

    def extract_entities(self, text: str) -> Dict:
        """
        Extract entities from text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with extracted entities
        """
        entities = {
            'people': [],
            'organizations': [],
            'companies': [],
            'locations': [],
            'tickers': [],
            'industries': []
        }

        # Use spaCy if available
        if self.spacy_model:
            entities = self._extract_with_spacy(text)
        else:
            entities = self._extract_basic(text)

        # Extract stock tickers
        entities['tickers'] = self._extract_tickers(text)

        # Map companies to tickers
        entities['company_ticker_map'] = self._map_companies_to_tickers(entities)

        return entities

    def _extract_with_spacy(self, text: str) -> Dict:
        """Extract entities using spaCy NER"""
        try:
            doc = self.spacy_model(text)

            entities = {
                'people': [],
                'organizations': [],
                'companies': [],
                'locations': [],
                'industries': []
            }

            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['people'].append(ent.text)
                elif ent.label_ == 'ORG':
                    # Distinguish between companies and other organizations
                    if self._is_company(ent.text):
                        entities['companies'].append(ent.text)
                    else:
                        entities['organizations'].append(ent.text)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['locations'].append(ent.text)

            # Deduplicate
            entities['people'] = list(set(entities['people']))
            entities['organizations'] = list(set(entities['organizations']))
            entities['companies'] = list(set(entities['companies']))
            entities['locations'] = list(set(entities['locations']))

            return entities

        except Exception as e:
            logger.error(f"Error in spaCy extraction: {e}")
            return self._extract_basic(text)

    def _extract_basic(self, text: str) -> Dict:
        """Basic entity extraction using patterns"""
        entities = {
            'people': [],
            'organizations': [],
            'companies': [],
            'locations': []
        }

        # Extract capitalized phrases (potential names)
        # This is a simple heuristic
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Extract known companies
        text_lower = text.lower()
        for company, ticker in self.company_tickers.items():
            if company in text_lower:
                entities['companies'].append(company.title())

        # Remove duplicates
        entities['companies'] = list(set(entities['companies']))

        return entities

    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock ticker symbols from text

        Args:
            text: Text to search

        Returns:
            List of potential ticker symbols
        """
        # Find patterns like $AAPL or just AAPL (all caps, 1-5 letters)
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
        word_tickers = re.findall(r'\b([A-Z]{2,5})\b', text)

        # Combine and filter
        all_tickers = set(dollar_tickers + word_tickers)

        # Filter out common English words that might match pattern
        common_words = {'USA', 'UK', 'CEO', 'CFO', 'IPO', 'ETF', 'NYSE', 'NASDAQ', 'SEC', 'FDA', 'FBI', 'CIA'}
        tickers = [t for t in all_tickers if t not in common_words]

        return list(set(tickers))

    def _map_companies_to_tickers(self, entities: Dict) -> Dict[str, str]:
        """
        Map company names to stock tickers

        Args:
            entities: Entities dictionary

        Returns:
            Dictionary mapping company name to ticker
        """
        mapping = {}

        for company in entities.get('companies', []):
            company_lower = company.lower()
            if company_lower in self.company_tickers:
                mapping[company] = self.company_tickers[company_lower]

        return mapping

    def _is_company(self, org_name: str) -> bool:
        """
        Determine if an organization is likely a company

        Args:
            org_name: Organization name

        Returns:
            True if likely a company
        """
        # Common company suffixes
        company_suffixes = ['Inc', 'Corp', 'Ltd', 'LLC', 'Co', 'Company', 'Corporation']

        for suffix in company_suffixes:
            if suffix in org_name:
                return True

        # Check against known companies
        if org_name.lower() in self.company_tickers:
            return True

        return False

    def identify_industry(self, text: str) -> List[str]:
        """
        Identify industries mentioned in text

        Args:
            text: Text to analyze

        Returns:
            List of identified industries
        """
        text_lower = text.lower()

        industry_keywords = {
            'technology': ['tech', 'software', 'ai', 'artificial intelligence', 'cloud', 'digital', 'cyber', 'semiconductor'],
            'financial': ['bank', 'finance', 'financial', 'investment', 'trading', 'credit', 'loan', 'payment'],
            'healthcare': ['health', 'medical', 'pharma', 'drug', 'hospital', 'biotech', 'medicine'],
            'energy': ['energy', 'oil', 'gas', 'renewable', 'power', 'electric', 'solar', 'wind'],
            'consumer': ['retail', 'consumer', 'shopping', 'ecommerce', 'store', 'brand'],
            'industrial': ['manufacturing', 'industrial', 'factory', 'production', 'machinery'],
            'real_estate': ['real estate', 'property', 'housing', 'construction', 'building'],
            'telecommunications': ['telecom', 'wireless', '5g', 'network', 'communication'],
            'automotive': ['auto', 'car', 'vehicle', 'automotive', 'electric vehicle', 'ev'],
            'aerospace': ['aerospace', 'aviation', 'airline', 'aircraft', 'space']
        }

        identified_industries = []

        for industry, keywords in industry_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    identified_industries.append(industry)
                    break

        return list(set(identified_industries))

    def calculate_entity_relevance(
        self,
        article_entities: Dict,
        target_companies: List[str] = None,
        target_industries: List[str] = None
    ) -> float:
        """
        Calculate relevance score based on entity matches

        Args:
            article_entities: Entities extracted from article
            target_companies: Companies to match against
            target_industries: Industries to match against

        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        max_score = 0.0

        # Direct company mentions (highest weight)
        if target_companies:
            max_score += 10.0
            article_companies = [c.lower() for c in article_entities.get('companies', [])]
            for target in target_companies:
                if target.lower() in article_companies:
                    score += 10.0
                    break

        # Ticker mentions
        if target_companies:
            max_score += 5.0
            article_tickers = article_entities.get('tickers', [])
            company_ticker_map = article_entities.get('company_ticker_map', {})

            for target in target_companies:
                target_lower = target.lower()
                if target_lower in self.company_tickers:
                    target_ticker = self.company_tickers[target_lower]
                    if target_ticker in article_tickers:
                        score += 5.0
                        break

        # Industry mentions
        if target_industries:
            max_score += 3.0
            article_industries = article_entities.get('industries', [])
            for target_industry in target_industries:
                if target_industry in article_industries:
                    score += 3.0
                    break

        # Normalize score
        if max_score > 0:
            return min(score / max_score, 1.0)
        else:
            return 0.0


if __name__ == "__main__":
    # Test the analyzer
    analyzer = EntityAnalyzer()

    test_text = """
    Apple Inc. CEO Tim Cook announced record iPhone sales today.
    The tech giant's stock (AAPL) surged 5% on the news.
    Microsoft and Google are also seeing strong performance in cloud services.
    """

    entities = analyzer.extract_entities(test_text)

    print("Extracted Entities:")
    print(f"People: {entities['people']}")
    print(f"Companies: {entities['companies']}")
    print(f"Tickers: {entities['tickers']}")
    print(f"Company-Ticker Map: {entities['company_ticker_map']}")

    industries = analyzer.identify_industry(test_text)
    print(f"\nIndustries: {industries}")
