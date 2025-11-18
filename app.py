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
    page_title="Topic Sentiment vs Stock Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stAlert {background-color: #1a1d29;}
    h1 {color: #00bfff; font-size: 2.5em;}
    h2 {color: #00ff88;}
    h3 {color: #ff6b35;}
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource
def get_database():
    return NewsStockDatabase()

db = get_database()

# Title
st.title("📊 Topic Sentiment vs Stock Price Dashboard")
st.markdown("**Search any topic** (Trump, Oil, Earthquake, Korea, etc.) and see how it correlates with stock movements")

st.markdown("---")

# Sidebar - API Keys & Fetch
st.sidebar.title("⚙️ Configuration")

with st.sidebar.expander("🔑 API Keys", expanded=False):
    newsdata_key = st.text_input("NewsData.io API Key", type="password",
                                 help="Free: https://newsdata.io/")

    if st.button("💾 Save API Key"):
        os.environ['NEWSDATA_API_KEY'] = newsdata_key
        st.success("✅ Saved!")

st.sidebar.markdown("---")

# Fetch News by Topic
st.sidebar.subheader("📰 Fetch News by Topic")

query_input = st.sidebar.text_input(
    "Search Topic/Query",
    placeholder="Trump, Oil, Europe, Earthquake, Gold...",
    help="Any keyword or topic"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("From Date", datetime.now() - timedelta(days=7))
with col2:
    end_date = st.date_input("To Date", datetime.now())

news_source = st.sidebar.selectbox("News Source", ["newsdata", "worldnews"])

if st.sidebar.button("🚀 Fetch News for Topic", type="primary"):
    if not query_input:
        st.sidebar.error("⚠️ Please enter a topic/query")
    else:
        # Check if already exists
        query_id = db.save_query(query_input, start_date.isoformat(), end_date.isoformat())

        if query_id is None:
            st.sidebar.warning(f"⚠️ **Duplicate!** Query '{query_input}' for {start_date} to {end_date} already exists in database.")
        else:
            with st.sidebar:
                with st.spinner(f"Fetching news about '{query_input}'..."):
                    # Fetch news
                    collector = NewsCollector(source=news_source)
                    articles = collector.fetch_news_by_query(
                        query_input,
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.max.time())
                    )

                    if articles:
                        # Analyze
                        sentiment_analyzer = SentimentAnalyzer(method='textblob')
                        entity_analyzer = EntityAnalyzer()
                        algorithm = NewsStockAlgorithm()

                        analyzed = algorithm.analyze_news_batch(
                            articles, sentiment_analyzer, entity_analyzer
                        )

                        # Save with query_id
                        db.save_articles_batch(analyzed, query_id)
                        db.update_query_article_count(query_id, len(analyzed))

                        st.success(f"✅ Fetched {len(analyzed)} articles for '{query_input}'!")
                        st.rerun()
                    else:
                        st.error("❌ No articles found. Check API key or try different dates.")

# Main Content
st.markdown("---")

# Get all queries
queries = db.get_all_queries()

if not queries:
    st.info("👆 **Get Started:** Enter a topic in the sidebar and click 'Fetch News'")
    st.markdown("""
    ### Examples:
    - **Trump** - Politics/Elections
    - **Oil** or **Gas** - Energy news
    - **Gold** - Commodities
    - **Europe** - European affairs
    - **Earthquake** or **Hurricane** - Natural disasters
    - **Korea** or **China** - Asian affairs
    - **AI** or **ChatGPT** - Technology
    """)
    st.stop()

# Query History
st.subheader("📚 Your Saved Queries")

queries_df = pd.DataFrame(queries)
queries_df['fetched_at'] = pd.to_datetime(queries_df['fetched_at']).dt.strftime('%Y-%m-%d %H:%M')

st.dataframe(
    queries_df[['query_text', 'start_date', 'end_date', 'article_count', 'fetched_at']],
    column_config={
        "query_text": "Query",
        "start_date": "From",
        "end_date": "To",
        "article_count": "Articles",
        "fetched_at": "Fetched At"
    },
    use_container_width=True,
    height=200
)

st.markdown("---")

# Analysis Section
st.subheader("📊 Topic Sentiment vs Stock Price")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_query_id = st.selectbox(
        "Select Topic/Query",
        options=[q['id'] for q in queries],
        format_func=lambda x: next(q['query_text'] for q in queries if q['id'] == x)
    )

with col2:
    # Stock symbol input
    stock_symbol = st.text_input("Stock Symbol", value="AAPL", help="e.g., AAPL, MSFT, TSLA")

with col3:
    if st.button("📈 Analyze"):
        st.rerun()

selected_query = next(q for q in queries if q['id'] == selected_query_id)

# Get articles for this query
cursor = db.conn.cursor()
cursor.execute("""
    SELECT publish_date, sentiment_polarity, sentiment_type, title, source, total_score
    FROM articles
    WHERE query_id = ?
    ORDER BY publish_date
""", (selected_query_id,))

query_articles = cursor.fetchall()

if not query_articles:
    st.warning(f"No articles found for query: {selected_query['query_text']}")
    st.stop()

# Aggregate by date
df_articles = pd.DataFrame(query_articles, columns=[
    'publish_date', 'sentiment_polarity', 'sentiment_type', 'title', 'source', 'total_score'
])
df_articles['publish_date'] = pd.to_datetime(df_articles['publish_date'])
df_articles['date'] = df_articles['publish_date'].dt.date

daily_sentiment = df_articles.groupby('date').agg({
    'sentiment_polarity': 'mean',
    'title': 'count'
}).reset_index()
daily_sentiment.columns = ['date', 'avg_sentiment', 'article_count']

# Get stock data
stock_collector = StockCollector()
stock_start = pd.to_datetime(selected_query['start_date'])
stock_end = pd.to_datetime(selected_query['end_date'])

stock_df = stock_collector.fetch_stock_data(
    stock_symbol.upper(),
    stock_start,
    stock_end
)

if stock_df is None or stock_df.empty:
    st.error(f"❌ Could not fetch stock data for {stock_symbol.upper()}")
    st.stop()

# Calculate returns
stock_df['Daily_Return'] = stock_df['Close'].pct_change() * 100

# Merge data
stock_df.index = pd.to_datetime(stock_df.index).tz_localize(None)
stock_df['date'] = stock_df.index.date

daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.date
merged = pd.merge(daily_sentiment, stock_df, on='date', how='outer').sort_values('date')
merged['date'] = pd.to_datetime(merged['date'])

# Calculate correlation
valid_data = merged.dropna(subset=['avg_sentiment', 'Daily_Return'])
if len(valid_data) > 2:
    correlation = valid_data['avg_sentiment'].corr(valid_data['Daily_Return'])
else:
    correlation = 0.0

# Display metrics
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.metric("📰 Topic", selected_query['query_text'])
with metric_col2:
    st.metric("📈 Stock", stock_symbol.upper())
with metric_col3:
    st.metric("📊 Correlation", f"{correlation:+.3f}")
with metric_col4:
    avg_sent = df_articles['sentiment_polarity'].mean()
    sent_emoji = "🟢" if avg_sent > 0 else "🔴" if avg_sent < 0 else "⚪"
    st.metric(f"{sent_emoji} Avg Sentiment", f"{avg_sent:+.3f}")

# BIG INTERACTIVE CHART
st.markdown("### 📊 Interactive Chart: Topic Sentiment vs Stock Price")

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    subplot_titles=(
        f'"{selected_query["query_text"]}" Sentiment vs {stock_symbol.upper()} Stock Price',
        'Daily News Volume'
    ),
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    vertical_spacing=0.12
)

# Sentiment line
fig.add_trace(
    go.Scatter(
        x=merged['date'],
        y=merged['avg_sentiment'],
        name=f'{selected_query["query_text"]} Sentiment',
        line=dict(color='#00bfff', width=4),
        mode='lines+markers',
        marker=dict(size=8)
    ),
    row=1, col=1, secondary_y=False
)

# Stock price line
fig.add_trace(
    go.Scatter(
        x=merged['date'],
        y=merged['Close'],
        name=f'{stock_symbol.upper()} Price',
        line=dict(color='#ff3366', width=4),
        mode='lines+markers',
        marker=dict(size=8)
    ),
    row=1, col=1, secondary_y=True
)

# Article volume bars
fig.add_trace(
    go.Bar(
        x=merged['date'],
        y=merged['article_count'],
        name='News Articles',
        marker_color='#00ff88',
        opacity=0.6
    ),
    row=2, col=1
)

# Layout
fig.update_layout(
    height=800,
    plot_bgcolor='#1a1a1a',
    paper_bgcolor='#0e1117',
    font=dict(color='#e0e0e0', size=12),
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=14)
    )
)

fig.update_xaxes(showgrid=True, gridcolor='#3d3d3d', title_font=dict(size=14))
fig.update_yaxes(showgrid=True, gridcolor='#3d3d3d', title_font=dict(size=14))

fig.update_yaxes(title_text="Sentiment Score", row=1, col=1, secondary_y=False,
                 title_font=dict(color='#00bfff', size=16))
fig.update_yaxes(title_text=f"Stock Price ($)", row=1, col=1, secondary_y=True,
                 title_font=dict(color='#ff3366', size=16))
fig.update_yaxes(title_text="Article Count", row=2, col=1,
                 title_font=dict(size=14))

st.plotly_chart(fig, use_container_width=True)

# Interpretation
st.markdown("### 💡 Interpretation")

if abs(correlation) > 0.5:
    strength = "**Strong**"
    color = "🟢"
elif abs(correlation) > 0.3:
    strength = "**Moderate**"
    color = "🟡"
else:
    strength = "**Weak**"
    color = "🔴"

direction = "positive" if correlation > 0 else "negative"

st.info(f"""
{color} **{strength} {direction} correlation ({correlation:+.3f})**

- When sentiment about **"{selected_query['query_text']}"** goes {'up' if correlation > 0 else 'down'},
  **{stock_symbol.upper()}** stock tends to go {'up' if correlation > 0 else 'down'}.
- Based on {len(df_articles)} articles from {selected_query['start_date']} to {selected_query['end_date']}
""")

# News Articles Table
st.markdown("---")
st.subheader(f"📰 Articles about '{selected_query['query_text']}'")

# Sort options
sort_by = st.selectbox(
    "Sort by",
    ["Date (Newest)", "Date (Oldest)", "Most Positive", "Most Negative", "Highest Score"]
)

if sort_by == "Date (Newest)":
    df_display = df_articles.sort_values('publish_date', ascending=False)
elif sort_by == "Date (Oldest)":
    df_display = df_articles.sort_values('publish_date', ascending=True)
elif sort_by == "Most Positive":
    df_display = df_articles.sort_values('sentiment_polarity', ascending=False)
elif sort_by == "Most Negative":
    df_display = df_articles.sort_values('sentiment_polarity', ascending=True)
else:  # Highest Score
    df_display = df_articles.sort_values('total_score', ascending=False)

# Display table
for idx, row in df_display.head(20).iterrows():
    sentiment_color = "🟢" if row['sentiment_type'] == 'positive' else "🔴" if row['sentiment_type'] == 'negative' else "⚪"

    with st.expander(f"{sentiment_color} {row['title']}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Source:** {row['source']}")
        with col2:
            st.write(f"**Date:** {row['publish_date'].strftime('%Y-%m-%d %H:%M')}")
        with col3:
            st.write(f"**Sentiment:** {row['sentiment_polarity']:+.3f}")

# Footer
st.markdown("---")
st.caption("📊 Topic Sentiment vs Stock Dashboard | Search any topic and overlay with any stock")
