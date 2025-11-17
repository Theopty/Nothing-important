# News-Stock Sentiment Analysis Algorithm

A flexible, AI-powered algorithm that correlates news sentiment with stock market movements. Analyze how news articles impact stock prices, experiment with different parameters, and improve your algorithm through trial and error.

## Features

- **Multi-Source News Collection**: Fetch news from NewsData.io (recommended) or WorldNewsAPI with fallback sample data
- **Stock Market Data**: Real-time stock data from yfinance covering indices, sectors, and individual stocks
- **AI-Powered Analysis**:
  - Sentiment analysis (TextBlob, FinBERT, or OpenAI)
  - Entity extraction (people, companies, organizations)
  - Industry classification
  - Framing analysis (positive/negative perspectives)
- **Flexible Algorithm Framework**: Easily adjust parameters and weights
- **Daily Aggregation**: Organize news by day to see multiple perspectives
- **Correlation Analysis**: Measure relationships between sentiment and stock movements
- **Experiment Tracking**: Compare different algorithm configurations
- **Visualization Tools**: Charts and reports for analysis results
- **Data Storage**: SQLite database for historical analysis

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the repository
cd Nothing-important

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Download spaCy model for better entity extraction
python -m spacy download en_core_web_sm
```

### 2. Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
NEWSDATA_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # Optional
```

Get API keys:
- **NewsData.io**: https://newsdata.io/ (FREE - 200 requests/day) ⭐ **RECOMMENDED**
- **WorldNewsAPI**: https://worldnewsapi.com/ (Alternative - 100 requests/day free tier)
- **OpenAI**: https://platform.openai.com/ (Optional, for enhanced sentiment analysis)

### 3. Run Your First Analysis

```bash
# Run with default configuration (uses sample data if no API key)
python main.py

# Run with specific symbols
python main.py --symbols AAPL MSFT GOOGL

# Run for a specific time period
python main.py --days 14 --symbols TSLA NVDA
```

## Usage Examples

### Basic Analysis

```bash
# Analyze specific stocks over the past 7 days
python main.py --symbols AAPL MSFT TSLA

# Analyze over 30 days
python main.py --days 30 --symbols NVDA AMD
```

### Running Experiments

The experiment framework helps you find the optimal parameters:

```bash
# Run weight optimization experiments
python experiment_runner.py --suite weights

# Run sentiment multiplier experiments
python experiment_runner.py --suite sentiment

# Run all experiment suites
python experiment_runner.py --suite all

# Run a custom experiment
python experiment_runner.py --custom experiments/example_custom_experiment.json
```

### Creating Custom Experiments

1. Copy the example experiment file:
```bash
cp experiments/example_custom_experiment.json experiments/my_experiment.json
```

2. Edit `my_experiment.json` with your parameters:
```json
{
  "name": "my_experiment",
  "description": "Testing increased sentiment weight",
  "config": {
    "weights": {
      "sentiment_score": 0.45,
      "entity_relevance": 0.25,
      "source_diversity": 0.15,
      "volume_spike": 0.10,
      "framing_consistency": 0.05
    }
  }
}
```

3. Run your experiment:
```bash
python experiment_runner.py --custom experiments/my_experiment.json
```

## Configuration

The `config.yaml` file controls all algorithm behavior:

### Key Parameters to Experiment With

#### Algorithm Weights (must sum to 1.0)
```yaml
algorithm:
  weights:
    sentiment_score: 0.35        # How much to trust sentiment
    entity_relevance: 0.25       # Importance of company mentions
    source_diversity: 0.15       # Multiple sources = more reliable
    volume_spike: 0.15           # Trading volume changes
    framing_consistency: 0.10    # How consistently news is framed
```

#### Scoring Parameters
```yaml
algorithm:
  scoring:
    positive_sentiment_multiplier: 1.2   # Boost for positive news
    negative_sentiment_multiplier: 1.5   # Boost for negative news (often stronger)
    direct_company_mention: 2.0          # Direct mention weight
    volume_spike_threshold: 1.5          # 150% of average = spike
```

### Tips for Parameter Tuning

1. **Start with the baseline** configuration and run an experiment
2. **Adjust one parameter at a time** to understand its impact
3. **Monitor the correlation metrics** - higher is better
4. **Track your experiments** - the system saves all results
5. **Consider market conditions** - different parameters work better in different environments

## Understanding the Output

### Analysis Results

```
Symbol: AAPL
================================================
Correlations:
  Sentiment vs Price:   +0.234
  Sentiment vs Volume:  +0.156
  Score vs Price:       +0.289
  Predictive (next day): +0.312

Trading Signal: BUY
Confidence: 65.0%
Reasoning:
  - Strong positive sentiment (0.45)
  - High article volume (12 articles)
  - Historical predictive correlation: 0.31
```

### Interpreting Correlations

- **Sentiment vs Price**: How well sentiment tracks with price changes
- **Predictive**: Can sentiment predict *next day* returns? (Most important!)
- **-1.0 to +1.0 range**:
  - Close to 0: No relationship
  - 0.3 to 0.5: Moderate relationship
  - 0.5+: Strong relationship

### Trading Signals

- **BUY**: Positive sentiment + good historical correlation
- **SELL**: Negative sentiment + good historical correlation
- **HOLD**: Unclear signal or insufficient data

**IMPORTANT**: These are experimental signals for research purposes only. Not financial advice!

## Project Structure

```
Nothing-important/
├── main.py                      # Main analysis runner
├── experiment_runner.py         # Experiment framework
├── config.yaml                  # Algorithm configuration
├── requirements.txt             # Python dependencies
├── .env                         # API keys (create from .env.example)
│
├── src/
│   ├── collectors/              # Data collection modules
│   │   ├── news_collector.py   # Fetch news from WorldNewsAPI
│   │   └── stock_collector.py  # Fetch stock data
│   ├── analyzers/               # AI analysis modules
│   │   ├── sentiment_analyzer.py   # Sentiment analysis
│   │   └── entity_analyzer.py      # Entity extraction
│   ├── algorithm/               # Core algorithm
│   │   └── news_stock_algorithm.py # Main algorithm logic
│   ├── storage/                 # Data persistence
│   │   └── database.py          # SQLite database
│   └── utils/                   # Utilities
│       └── visualizer.py        # Visualization tools
│
├── data/                        # Database and cache (created on first run)
├── experiments/                 # Experiment results
│   ├── configs/                 # Custom experiment configs
│   └── results/                 # Experiment output
└── reports/                     # Generated reports and charts
```

## How It Works

### 1. Data Collection
- Fetches news articles from WorldNewsAPI (or uses sample data)
- Collects stock market data from yfinance
- Organizes data by date for daily analysis

### 2. AI Analysis
Each article is analyzed for:
- **Sentiment**: Positive, negative, or neutral (with confidence score)
- **Entities**: Companies, people, organizations mentioned
- **Industries**: Technology, finance, healthcare, etc.
- **Framing**: How the news is presented

### 3. Scoring
The algorithm combines multiple factors:
- Sentiment strength and confidence
- Entity relevance (company mentions)
- Source diversity (multiple sources = more reliable)
- Volume spikes (unusual trading activity)
- Framing consistency (how uniformly news is presented)

### 4. Correlation
- Aggregates sentiment by day
- Compares with stock price movements
- Calculates correlation coefficients
- Tests predictive power (can sentiment predict next-day returns?)

### 5. Signal Generation
Based on:
- Current sentiment
- Historical correlation
- Article volume
- Confidence metrics

## Advanced Features

### Database Queries

The SQLite database stores all data for historical analysis:

```python
from src.storage.database import NewsStockDatabase

db = NewsStockDatabase()

# Get articles from a date range
articles = db.get_articles_by_date('2025-01-01', '2025-01-31')

# Get stock data
stock_df = db.get_stock_data('AAPL', '2025-01-01', '2025-01-31')

# Get experiment history
experiments = db.get_experiments(limit=10)
```

### Custom Analysis

You can use individual components for custom analysis:

```python
from src.collectors.news_collector import NewsCollector
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from datetime import datetime, timedelta

# Collect news
collector = NewsCollector()
articles = collector.fetch_news(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)

# Analyze sentiment
analyzer = SentimentAnalyzer(method='textblob')
for article in articles:
    sentiment = analyzer.analyze(article['text'])
    print(f"{article['title']}: {sentiment['sentiment']}")
```

### Visualization

Create custom visualizations:

```python
from src.utils.visualizer import Visualizer

viz = Visualizer()

# Plot sentiment timeline
viz.plot_sentiment_timeline(daily_aggregates, 'AAPL')

# Plot correlations
viz.plot_correlation_analysis(daily_aggregates, stock_data, 'AAPL')

# Compare experiments
viz.plot_experiment_comparison(experiments)
```

## Improving Your Algorithm

### Iteration Process

1. **Run baseline experiment**
   ```bash
   python experiment_runner.py --suite weights
   ```

2. **Analyze results** - Look for:
   - Which configuration had highest correlation?
   - Which signals were most accurate?
   - Where did the algorithm fail?

3. **Form hypothesis** - Example:
   - "Negative news seems to have stronger impact"
   - "Multiple sources increase reliability"
   - "Technology stocks respond faster to news"

4. **Create custom experiment** to test your hypothesis

5. **Compare results** with previous experiments

6. **Iterate** - Repeat the process

### Common Improvements

- **Increase negative sentiment multiplier** - Negative news often has stronger impact
- **Add industry-specific weights** - Different sectors respond differently
- **Time decay** - Recent news matters more
- **Source credibility** - Weight trusted sources higher
- **Volume confirmation** - Require volume spike for signals

### Example: Industry-Specific Configuration

Create `experiments/tech_focused.json`:
```json
{
  "name": "tech_sector_optimized",
  "description": "Optimized for technology stocks",
  "config": {
    "weights": {
      "sentiment_score": 0.40,
      "entity_relevance": 0.30,
      "source_diversity": 0.20,
      "volume_spike": 0.10
    },
    "scoring": {
      "negative_sentiment_multiplier": 2.0,
      "direct_company_mention": 2.5
    }
  }
}
```

## Troubleshooting

### No news articles found
- Check your WorldNewsAPI key in `.env`
- The system will use sample data if API is unavailable
- Free tier has rate limits - try reducing date range

### Stock data errors
- yfinance might be temporarily unavailable
- Check symbol spelling (use UPPERCASE: AAPL not aapl)
- Some symbols might not have data for your date range

### Low correlations
This is normal! Real-world correlations between news and stocks are typically low:
- 0.2-0.4 is actually pretty good
- Perfect correlation (0.8+) is suspicious
- Focus on **predictive** correlation (next-day returns)

### Python dependency issues
```bash
# Upgrade pip
pip install --upgrade pip

# Install specific versions
pip install pandas==2.1.0 yfinance==0.2.32
```

## FAQ

**Q: Is this financial advice?**
A: No! This is an experimental research tool for learning about sentiment analysis and market correlation.

**Q: Can I use this for real trading?**
A: This is designed for research and learning. Real trading requires much more sophisticated systems, risk management, and regulatory compliance.

**Q: Why are correlations low?**
A: Markets are complex with many factors. News sentiment is just one factor. Low correlations (0.2-0.4) are actually typical.

**Q: How do I get better results?**
A: Experiment with parameters, use more data, focus on specific sectors, and combine with other indicators.

**Q: Can I add other news sources?**
A: Yes! Extend `NewsCollector` class to add new sources. Pull requests welcome!

**Q: What about intraday trading?**
A: The system supports intraday data - check `get_intraday_data()` in `stock_collector.py`.

## Contributing

This is a flexible framework designed for experimentation. Ideas for enhancement:

- [ ] Additional news sources (RSS, Twitter, etc.)
- [ ] More sophisticated ML models
- [ ] Real-time streaming data
- [ ] Backtesting framework
- [ ] Risk management systems
- [ ] More visualization options

## License

This project is open source - use for research and learning.

## Disclaimer

This tool is for educational and research purposes only. It does not provide financial advice. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.

## Support

For issues, questions, or suggestions:
1. Check this README
2. Review the example configurations
3. Examine the experiment results
4. Try different parameters

Happy experimenting! 🚀📈
