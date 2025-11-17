# System Architecture

## Overview

The News-Stock Sentiment Analysis Algorithm is designed as a modular, extensible system for correlating news sentiment with stock market movements.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Runner                           │
│                      (main.py)                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│   Collectors  │   │   Analyzers   │
├───────────────┤   ├───────────────┤
│ NewsCollector │   │ Sentiment     │
│ StockCollector│   │ Entity        │
└───────┬───────┘   └───────┬───────┘
        │                   │
        └─────────┬─────────┘
                  ▼
        ┌─────────────────┐
        │    Algorithm    │
        │  (Scoring &     │
        │  Correlation)   │
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│   Storage    │  │ Visualization│
│  (Database)  │  │   (Charts)   │
└──────────────┘  └──────────────┘
```

## Component Details

### 1. Data Collection Layer

#### NewsCollector (`src/collectors/news_collector.py`)
- **Purpose**: Fetch news articles from external APIs
- **Primary Source**: WorldNewsAPI
- **Features**:
  - Date range queries
  - Category filtering
  - Keyword search
  - Sample data fallback
- **Output**: List of article dictionaries

#### StockCollector (`src/collectors/stock_collector.py`)
- **Purpose**: Fetch stock market data
- **Source**: yfinance
- **Features**:
  - Historical price data
  - Multiple symbols
  - Intraday data support
  - Volume spike detection
  - Technical indicators (MA, volatility)
- **Output**: pandas DataFrame

### 2. Analysis Layer

#### SentimentAnalyzer (`src/analyzers/sentiment_analyzer.py`)
- **Purpose**: Analyze sentiment of news articles
- **Methods**:
  - TextBlob (default, no API required)
  - FinBERT (financial domain-specific)
  - OpenAI GPT (most sophisticated)
- **Output**:
  ```python
  {
    'sentiment': 'positive/negative/neutral',
    'polarity': -1.0 to +1.0,
    'confidence': 0.0 to 1.0,
    'method': 'textblob/finbert/openai'
  }
  ```

#### EntityAnalyzer (`src/analyzers/entity_analyzer.py`)
- **Purpose**: Extract entities and classify industries
- **Methods**:
  - spaCy NER (when available)
  - Pattern matching fallback
  - Ticker symbol extraction
  - Company-ticker mapping
- **Output**:
  ```python
  {
    'people': [...],
    'organizations': [...],
    'companies': [...],
    'tickers': [...],
    'industries': [...]
  }
  ```

### 3. Algorithm Layer

#### NewsStockAlgorithm (`src/algorithm/news_stock_algorithm.py`)
- **Purpose**: Core scoring and correlation logic
- **Configuration-Driven**: All parameters in `config.yaml`

**Key Methods**:

1. **analyze_news_batch()**
   - Processes multiple articles
   - Applies sentiment and entity analysis
   - Calculates component scores

2. **_calculate_sentiment_score()**
   - Converts sentiment to normalized score
   - Applies multipliers (positive/negative)
   - Weight by confidence

3. **_calculate_entity_score()**
   - Scores entity relevance
   - Company mentions → higher score
   - Multiple entities → reliability bonus

4. **aggregate_daily_sentiment()**
   - Groups articles by date
   - Calculates statistics
   - Identifies top articles

5. **correlate_with_stock()**
   - Merges news and stock data
   - Calculates correlation coefficients
   - Identifies significant events

6. **generate_trading_signals()**
   - BUY/SELL/HOLD recommendations
   - Confidence scores
   - Reasoning explanation

### 4. Storage Layer

#### NewsStockDatabase (`src/storage/database.py`)
- **Technology**: SQLite
- **Purpose**: Persistent storage for analysis and experiments

**Schema**:

```sql
-- News articles with analysis results
articles (
  id, title, text, sentiment_polarity, sentiment_type,
  entities_json, industries_json, sentiment_score,
  entity_score, total_score, analyzed_at
)

-- Stock market data
stock_data (
  symbol, date, open, high, low, close, volume,
  daily_return, volume_change
)

-- Daily aggregated sentiment
daily_aggregates (
  date, symbol, article_count, average_sentiment,
  sentiment_distribution, source_count
)

-- Correlation analysis results
correlations (
  symbol, period_start, period_end,
  sentiment_price_corr, predictive_corr
)

-- Experiment tracking
experiments (
  experiment_name, version, config_json,
  results_json, metrics_json
)
```

### 5. Experiment Framework

#### ExperimentRunner (`experiment_runner.py`)
- **Purpose**: A/B testing of algorithm parameters
- **Features**:
  - Parameter sweep
  - Result comparison
  - Metric tracking
  - Automatic saving

**Experiment Types**:
- Weight optimization
- Sentiment multiplier tuning
- Custom parameter testing

### 6. Visualization Layer

#### Visualizer (`src/utils/visualizer.py`)
- **Purpose**: Generate charts and reports
- **Libraries**: matplotlib, seaborn, pandas

**Outputs**:
- Sentiment timeline charts
- Correlation scatter plots
- Experiment comparisons
- HTML summary reports

## Data Flow

### Standard Analysis Flow

```
1. User runs main.py
   ↓
2. Load config.yaml
   ↓
3. Initialize components (collectors, analyzers, algorithm)
   ↓
4. Collect news articles (NewsCollector)
   ↓
5. Analyze each article (SentimentAnalyzer, EntityAnalyzer)
   ↓
6. Score articles (NewsStockAlgorithm)
   ↓
7. Collect stock data (StockCollector)
   ↓
8. Aggregate by day (NewsStockAlgorithm)
   ↓
9. Calculate correlations (NewsStockAlgorithm)
   ↓
10. Generate signals (NewsStockAlgorithm)
    ↓
11. Save to database (NewsStockDatabase)
    ↓
12. Display results to user
```

### Experiment Flow

```
1. User runs experiment_runner.py
   ↓
2. Define experiment parameters
   ↓
3. For each experiment configuration:
   ├─ Create custom config
   ├─ Run standard analysis
   ├─ Calculate metrics
   └─ Save results
   ↓
4. Compare all experiments
   ↓
5. Identify best performer
   ↓
6. Generate comparison charts
```

## Configuration System

### Hierarchical Configuration

```yaml
config.yaml (base configuration)
   ↓
experiment config (overrides)
   ↓
runtime parameters (command line args)
```

### Key Configuration Sections

1. **News Collection**
   - Sources
   - Categories
   - Date ranges
   - Volume limits

2. **Stock Market**
   - Symbols/Tickers
   - Indices
   - Sector ETFs
   - Data intervals

3. **AI Analysis**
   - Sentiment model selection
   - Entity extraction settings
   - Confidence thresholds

4. **Algorithm Weights** (most important!)
   - Component importance
   - Must sum to 1.0
   - Experimental tuning

5. **Scoring Rules**
   - Multipliers
   - Thresholds
   - Bonuses/Penalties

## Extensibility Points

### Adding New News Sources

```python
class CustomNewsCollector(NewsCollector):
    def fetch_from_custom_source(self, ...):
        # Implement your source
        return articles
```

### Adding New Sentiment Models

```python
class SentimentAnalyzer:
    def _analyze_custom_model(self, text):
        # Implement your model
        return sentiment_dict
```

### Adding New Scoring Factors

```python
class NewsStockAlgorithm:
    def _calculate_custom_score(self, article):
        # Implement your scoring logic
        return score
```

### Adding New Visualizations

```python
class Visualizer:
    def plot_custom_analysis(self, data):
        # Create your visualization
        plt.savefig('custom_chart.png')
```

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**: Articles analyzed in batches
2. **Database Indexing**: Quick lookups on symbol+date
3. **Caching**: Sample data cached in memory
4. **Lazy Loading**: Models loaded only when needed
5. **Parallel Collection**: News and stock data fetched concurrently (when possible)

### Scalability

- **Small Scale** (< 100 articles/day): Current architecture works well
- **Medium Scale** (100-1000 articles/day): Consider async processing
- **Large Scale** (1000+ articles/day): Consider:
  - Message queue (RabbitMQ, Kafka)
  - Distributed processing (Celery)
  - Time-series database (InfluxDB)
  - Caching layer (Redis)

## Security Considerations

1. **API Keys**: Stored in .env (gitignored)
2. **SQL Injection**: Parameterized queries used throughout
3. **Input Validation**: Date ranges, symbols validated
4. **Rate Limiting**: Respect API rate limits
5. **Data Privacy**: No PII stored

## Testing Strategy

### Unit Tests
- Test individual components
- Mock external APIs
- Verify calculations

### Integration Tests
- Test component interactions
- Use sample data
- Verify data flow

### Experiment Tests
- Baseline experiments
- Parameter validation
- Metric calculations

## Future Enhancements

### Planned Features
- [ ] Real-time streaming data
- [ ] Multiple sentiment models (ensemble)
- [ ] Social media integration (Twitter, Reddit)
- [ ] Machine learning predictions
- [ ] Backtesting framework
- [ ] Risk management module
- [ ] Portfolio optimization
- [ ] Alert system

### Architecture Evolution
- Microservices for scalability
- API gateway for external access
- Event-driven architecture
- Cloud deployment (AWS, GCP)

## Troubleshooting

### Common Issues

1. **Import Errors**: Check virtual environment activation
2. **API Errors**: Verify API keys in .env
3. **Data Quality**: Use sample data to test logic
4. **Performance**: Reduce date range or article count

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

This architecture provides:
- **Modularity**: Easy to extend and modify
- **Configurability**: No code changes for experiments
- **Testability**: Clear component boundaries
- **Maintainability**: Well-documented and organized
- **Scalability**: Can grow with your needs

The system is designed for learning and experimentation. Start simple, then enhance!
