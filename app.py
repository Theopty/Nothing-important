import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

from src.storage.database import NewsStockDatabase
from src.collectors.news_collector import NewsCollector
from src.collectors.stock_collector import StockCollector
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.entity_analyzer import EntityAnalyzer
from src.algorithm.news_stock_algorithm import NewsStockAlgorithm

# Page config
st.set_page_config(
    page_title="News-Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1a1d29; padding: 15px; border-radius: 10px;}
    h1, h2, h3 {color: #00bfff;}
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_database():
    return NewsStockDatabase()

db = get_database()

# Sidebar - Configuration
st.sidebar.title("⚙️ Configuration")

# API Keys Section
with st.sidebar.expander("🔑 API Keys", expanded=False):
    newsdata_key = st.text_input("NewsData.io API Key", type="password",
                                 help="Get free key: https://newsdata.io/")
    openai_key = st.text_input("OpenAI API Key (Optional)", type="password")

    if st.button("Save API Keys"):
        # Save to environment
        os.environ['NEWSDATA_API_KEY'] = newsdata_key
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
        st.success("✅ API Keys saved!")

# Fetch News Section
with st.sidebar.expander("📰 Fetch News", expanded=True):
    fetch_symbols = st.text_input("Stock Symbols", "AAPL,MSFT,GOOGL",
                                  help="Comma-separated list")
    fetch_days = st.slider("Days to fetch", 1, 30, 7)
    news_source = st.selectbox("News Source", ["newsdata", "worldnews"])

    if st.button("🚀 Fetch & Analyze News"):
        with st.spinner("Fetching news and analyzing..."):
            symbols = [s.strip().upper() for s in fetch_symbols.split(',')]

            # Initialize collectors
            news_collector = NewsCollector(source=news_source)
            stock_collector = StockCollector()
            sentiment_analyzer = SentimentAnalyzer(method='textblob')
            entity_analyzer = EntityAnalyzer()
            algorithm = NewsStockAlgorithm()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=fetch_days)

            # Fetch news
            articles = news_collector.fetch_news(start_date, end_date)

            # Analyze articles
            analyzed_articles = algorithm.analyze_news_batch(
                articles, sentiment_analyzer, entity_analyzer
            )

            # Save to database
            db.save_articles_batch(analyzed_articles)

            # Fetch and save stock data
            for symbol in symbols:
                stock_df = stock_collector.fetch_stock_data(symbol, start_date, end_date)
                if stock_df is not None:
                    db.save_stock_data(symbol, stock_df)

                # Calculate daily aggregates
                for single_date in pd.date_range(start_date, end_date):
                    daily_agg = algorithm.aggregate_daily_sentiment(analyzed_articles, single_date)
                    daily_agg['symbol'] = symbol
                    db.save_daily_aggregate(daily_agg)

            st.success(f"✅ Fetched {len(analyzed_articles)} articles for {len(symbols)} symbols!")
            st.rerun()

# Main Dashboard
st.title("📊 News-Stock Sentiment Analysis Dashboard")

# Top metrics
col1, col2, col3, col4 = st.columns(4)

# Get summary stats
cursor = db.conn.cursor()
cursor.execute("SELECT COUNT(*) FROM articles")
total_articles = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_aggregates")
total_symbols = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(DISTINCT source) FROM articles")
total_sources = cursor.fetchone()[0]

cursor.execute("SELECT AVG(sentiment_polarity) FROM articles")
avg_sentiment_result = cursor.fetchone()
avg_sentiment = avg_sentiment_result[0] if avg_sentiment_result[0] is not None else 0

with col1:
    st.metric("📰 Total Articles", f"{total_articles:,}")
with col2:
    st.metric("📈 Symbols Tracked", total_symbols)
with col3:
    st.metric("📡 News Sources", total_sources)
with col4:
    sentiment_color = "🟢" if avg_sentiment > 0 else "🔴" if avg_sentiment < 0 else "⚪"
    st.metric(f"{sentiment_color} Avg Sentiment", f"{avg_sentiment:+.3f}")

st.markdown("---")

# Filters Section
st.subheader("🔍 Filters")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    # Get available symbols
    cursor.execute("SELECT DISTINCT symbol FROM daily_aggregates ORDER BY symbol")
    available_symbols = [row[0] for row in cursor.fetchall()]
    selected_symbol = st.selectbox("Stock Symbol", available_symbols if available_symbols else ['AAPL'])

with filter_col2:
    # Get available sources
    cursor.execute("SELECT DISTINCT source FROM articles ORDER BY source")
    sources = ['All'] + [row[0] for row in cursor.fetchall()]
    selected_source = st.selectbox("News Source", sources)

with filter_col3:
    # Get available entities/companies
    cursor.execute("SELECT DISTINCT entities_json FROM articles WHERE entities_json IS NOT NULL LIMIT 100")
    all_entities = set()
    for row in cursor.fetchall():
        try:
            import json
            entities = json.loads(row[0])
            all_entities.update(entities.get('companies', []))
        except:
            pass
    entities_list = ['All'] + sorted(list(all_entities))[:50]  # Top 50
    selected_entity = st.selectbox("Entity/Company", entities_list)

with filter_col4:
    date_range = st.slider(
        "Days to show",
        1, 90, 30
    )

# Get filtered data
end_date = datetime.now()
start_date = end_date - timedelta(days=date_range)

# Get daily aggregates
cursor.execute("""
    SELECT date, average_sentiment, article_count, positive_count, negative_count
    FROM daily_aggregates
    WHERE symbol = ? AND date BETWEEN ? AND ?
    ORDER BY date
""", (selected_symbol, start_date.date().isoformat(), end_date.date().isoformat()))

daily_data = cursor.fetchall()

# Get stock data
stock_df = db.get_stock_data(
    selected_symbol,
    start_date.date().isoformat(),
    end_date.date().isoformat()
)

if not stock_df.empty:
    stock_df.columns = stock_df.columns.str.capitalize()
    stock_df['Daily_Return'] = stock_df['Close'].pct_change() * 100

# Get filtered articles
article_query = """
    SELECT title, sentiment_polarity, sentiment_type, source, publish_date, total_score
    FROM articles
    WHERE publish_date BETWEEN ? AND ?
"""
params = [start_date.isoformat(), end_date.isoformat()]

if selected_source != 'All':
    article_query += " AND source = ?"
    params.append(selected_source)

if selected_entity != 'All':
    article_query += " AND entities_json LIKE ?"
    params.append(f'%{selected_entity}%')

article_query += " ORDER BY publish_date DESC"

cursor.execute(article_query, params)
filtered_articles = cursor.fetchall()

# Big Interactive Chart
st.subheader(f"📈 {selected_symbol} - Sentiment vs Stock Price")

if daily_data and not stock_df.empty:
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f'{selected_symbol} - Sentiment & Price Correlation',
            'Daily Article Volume'
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        vertical_spacing=0.1
    )

    # Prepare data
    dates = [datetime.strptime(d[0], '%Y-%m-%d') for d in daily_data]
    sentiments = [d[1] for d in daily_data]
    article_counts = [d[2] for d in daily_data]

    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=dates, y=sentiments,
            name='Sentiment',
            line=dict(color='#00bfff', width=3),
            mode='lines+markers'
        ),
        row=1, col=1, secondary_y=False
    )

    # Add stock price line
    fig.add_trace(
        go.Scatter(
            x=stock_df.index, y=stock_df['Close'],
            name='Stock Price',
            line=dict(color='#ff3366', width=3),
            mode='lines+markers'
        ),
        row=1, col=1, secondary_y=True
    )

    # Add article volume bars
    fig.add_trace(
        go.Bar(
            x=dates, y=article_counts,
            name='Articles',
            marker_color='#ff6b35',
            opacity=0.7
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=700,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#0e1117',
        font=dict(color='#e0e0e0'),
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=True, gridcolor='#3d3d3d')
    fig.update_yaxes(showgrid=True, gridcolor='#3d3d3d')

    # Y-axis labels
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Article Count", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Correlation metric
    if len(daily_data) > 1:
        df_merged = pd.DataFrame(daily_data, columns=['date', 'sentiment', 'articles', 'pos', 'neg'])
        df_merged['date'] = pd.to_datetime(df_merged['date'])
        df_merged.set_index('date', inplace=True)

        stock_df.index = pd.to_datetime(stock_df.index).tz_localize(None)
        merged = df_merged.join(stock_df[['Close', 'Daily_Return']], how='inner')

        if len(merged) > 2:
            corr = merged['sentiment'].corr(merged['Daily_Return'])

            st.info(f"📊 **Correlation: {corr:+.3f}** | "
                   f"Articles: {len(filtered_articles)} | "
                   f"Date Range: {start_date.date()} to {end_date.date()}")
else:
    st.warning(f"⚠️ No data found for {selected_symbol}. Click 'Fetch & Analyze News' in the sidebar.")

# News Articles Table
st.subheader(f"📰 News Articles ({len(filtered_articles)} found)")

if filtered_articles:
    # Convert to dataframe for display
    articles_df = pd.DataFrame(filtered_articles, columns=[
        'Title', 'Sentiment Score', 'Sentiment', 'Source', 'Date', 'Impact Score'
    ])

    # Format dates
    articles_df['Date'] = pd.to_datetime(articles_df['Date']).dt.strftime('%Y-%m-%d %H:%M')

    # Color code sentiment
    def color_sentiment(val):
        if val == 'positive':
            return 'background-color: #1a4d2e; color: #00ff88'
        elif val == 'negative':
            return 'background-color: #4d1a1a; color: #ff3366'
        else:
            return 'background-color: #2d2d2d'

    styled_df = articles_df.style.applymap(color_sentiment, subset=['Sentiment'])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

    # Download button
    csv = articles_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Articles as CSV",
        data=csv,
        file_name=f"{selected_symbol}_articles_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info("No articles found with current filters. Try adjusting your filters or fetch more news.")

# Source Impact Analysis (if enough data)
if len(filtered_articles) > 10:
    st.markdown("---")
    st.subheader(f"📡 Source Impact Analysis for {selected_symbol}")

    # Group by source
    cursor.execute("""
        SELECT source, COUNT(*) as count, AVG(sentiment_polarity) as avg_sentiment
        FROM articles
        WHERE publish_date BETWEEN ? AND ?
        GROUP BY source
        HAVING count > 2
        ORDER BY count DESC
        LIMIT 10
    """, (start_date.isoformat(), end_date.isoformat()))

    source_stats = cursor.fetchall()

    if source_stats:
        source_df = pd.DataFrame(source_stats, columns=['Source', 'Articles', 'Avg Sentiment'])

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart of article counts
            fig_sources = go.Figure(data=[
                go.Bar(
                    x=source_df['Source'],
                    y=source_df['Articles'],
                    marker_color='#00bfff',
                    text=source_df['Articles'],
                    textposition='auto'
                )
            ])
            fig_sources.update_layout(
                title="Articles per Source",
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#0e1117',
                font=dict(color='#e0e0e0'),
                height=400
            )
            st.plotly_chart(fig_sources, use_container_width=True)

        with col2:
            # Bar chart of average sentiment
            colors = ['#00ff88' if x > 0 else '#ff3366' for x in source_df['Avg Sentiment']]
            fig_sentiment = go.Figure(data=[
                go.Bar(
                    x=source_df['Source'],
                    y=source_df['Avg Sentiment'],
                    marker_color=colors,
                    text=[f"{x:+.2f}" for x in source_df['Avg Sentiment']],
                    textposition='auto'
                )
            ])
            fig_sentiment.update_layout(
                title="Average Sentiment by Source",
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#0e1117',
                font=dict(color='#e0e0e0'),
                height=400
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

# Footer
st.markdown("---")
st.caption("📊 News-Stock Sentiment Analysis Dashboard | Built with Streamlit")
