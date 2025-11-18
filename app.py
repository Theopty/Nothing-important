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

# Show existing queries for this topic
if query_input:
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT start_date, end_date, article_count, fetched_at
        FROM queries
        WHERE query_text = ?
        ORDER BY fetched_at DESC
    """, (query_input,))
    existing_queries = cursor.fetchall()

    if existing_queries:
        st.sidebar.info(f"📋 Found {len(existing_queries)} existing searches for '{query_input}':")
        for eq in existing_queries[:3]:  # Show last 3
            st.sidebar.caption(f"  • {eq['start_date']} to {eq['end_date']} ({eq['article_count']} articles)")

if st.sidebar.button("🚀 Fetch News for Topic", type="primary"):
    if not query_input:
        st.sidebar.error("⚠️ Please enter a topic/query")
    elif not newsdata_key and news_source == 'newsdata':
        st.sidebar.error("⚠️ Please enter your NewsData.io API key above first!")
    else:
        # Check if already exists
        query_id = db.save_query(query_input, start_date.isoformat(), end_date.isoformat())

        if query_id is None:
            st.sidebar.warning(f"⚠️ Already fetched! This exact query exists. Try different dates or analyze the existing data below.")
            # Don't block - they can still use the existing data
        else:
            with st.sidebar:
                with st.spinner(f"Fetching news about '{query_input}'..."):
                    # Fetch news - pass API key from UI
                    api_key = newsdata_key if news_source == 'newsdata' else None
                    collector = NewsCollector(api_key=api_key, source=news_source)

                    st.info(f"📡 Using {news_source} API...")

                    articles = collector.fetch_news_by_query(
                        query_input,
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.max.time())
                    )

                    if articles:
                        # Analyze
                        st.info(f"🤖 Analyzing {len(articles)} articles...")
                        sentiment_analyzer = SentimentAnalyzer(method='textblob')
                        entity_analyzer = EntityAnalyzer()
                        algorithm = NewsStockAlgorithm()

                        analyzed = algorithm.analyze_news_batch(
                            articles, sentiment_analyzer, entity_analyzer
                        )

                        # Save with query_id
                        st.info(f"💾 Saving to database...")
                        db.save_articles_batch(analyzed, query_id)
                        db.update_query_article_count(query_id, len(analyzed))

                        st.success(f"✅ Fetched {len(analyzed)} articles for '{query_input}'!")
                        st.rerun()
                    else:
                        st.error("❌ No articles found. Check the logs below for details.")
                        st.caption("Common issues:")
                        st.caption("• Invalid API key")
                        st.caption("• No news matching your query in the date range")
                        st.caption("• API rate limit reached")
                        st.caption("• Try broader date range or different query")

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
col_title, col_manage = st.columns([3, 1])
with col_title:
    st.subheader("📚 Your Saved Queries")
with col_manage:
    with st.expander("🗑️ Manage"):
        st.caption("Delete queries you no longer need")
        delete_query = st.selectbox(
            "Select query to delete:",
            options=[q['id'] for q in queries],
            format_func=lambda x: next(f"{q['query_text']} ({q['start_date']} to {q['end_date']})" for q in queries if q['id'] == x),
            key="delete_select"
        )
        if st.button("Delete Selected", type="secondary", key="delete_btn"):
            cursor = db.conn.cursor()
            # Delete articles first
            cursor.execute("DELETE FROM articles WHERE query_id = ?", (delete_query,))
            # Delete query
            cursor.execute("DELETE FROM queries WHERE id = ?", (delete_query,))
            db.conn.commit()
            st.success("✅ Deleted!")
            st.rerun()

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

# Analysis Mode Selection
analysis_mode = st.radio(
    "Analysis Mode",
    ["Single Topic Analysis", "Cross-Reference Multiple Topics"],
    horizontal=True,
    help="Single: Analyze one topic | Cross-Reference: Compare multiple topics together"
)

st.markdown("---")

if analysis_mode == "Single Topic Analysis":
    # Original single topic analysis
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
else:
    # Cross-Reference Analysis
    st.subheader("🔀 Cross-Reference Multiple Topics")
    st.markdown("Compare how multiple topics interact and find patterns in stock movements")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_query_ids = st.multiselect(
            "Select Topics to Cross-Reference",
            options=[q['id'] for q in queries],
            default=[q['id'] for q in queries][:min(2, len(queries))],
            format_func=lambda x: next(q['query_text'] for q in queries if q['id'] == x),
            help="Select 2 or more topics to compare (e.g., Trump + Europe)"
        )

    with col2:
        stock_symbol = st.text_input("Stock Symbol", value="AAPL", help="e.g., AAPL, MSFT, TSLA")

    if len(selected_query_ids) < 2:
        st.warning("⚠️ Please select at least 2 topics to cross-reference")
        st.stop()

    # Fetch data for all selected queries
    cross_ref_data = {}
    all_sentiment_data = []

    for query_id in selected_query_ids:
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT publish_date, sentiment_polarity, sentiment_type, title, source
            FROM articles
            WHERE query_id = ?
            ORDER BY publish_date
        """, (query_id,))

        query_articles = cursor.fetchall()
        if not query_articles:
            continue

        query_info = next(q for q in queries if q['id'] == query_id)

        df_query = pd.DataFrame(query_articles, columns=[
            'publish_date', 'sentiment_polarity', 'sentiment_type', 'title', 'source'
        ])
        df_query['publish_date'] = pd.to_datetime(df_query['publish_date'])
        df_query['date'] = df_query['publish_date'].dt.date

        daily = df_query.groupby('date').agg({
            'sentiment_polarity': 'mean',
            'title': 'count'
        }).reset_index()
        daily.columns = ['date', 'avg_sentiment', 'article_count']
        daily['date'] = pd.to_datetime(daily['date'])
        daily['query_text'] = query_info['query_text']
        daily['query_id'] = query_id

        cross_ref_data[query_id] = {
            'query_text': query_info['query_text'],
            'daily': daily,
            'articles': df_query
        }

        all_sentiment_data.append(daily)

    if len(cross_ref_data) < 2:
        st.error("Not enough data for selected topics")
        st.stop()

    # Get stock data
    stock_collector = StockCollector()
    all_dates = pd.concat([d['daily']['date'] for d in cross_ref_data.values()])
    stock_start = all_dates.min()
    stock_end = all_dates.max()

    stock_df = stock_collector.fetch_stock_data(
        stock_symbol.upper(),
        stock_start,
        stock_end
    )

    if stock_df is None or stock_df.empty:
        st.error(f"❌ Could not fetch stock data for {stock_symbol.upper()}")
        st.stop()

    stock_df['Daily_Return'] = stock_df['Close'].pct_change() * 100
    stock_df.index = pd.to_datetime(stock_df.index).tz_localize(None)
    stock_df['date'] = stock_df.index.date
    stock_df['date'] = pd.to_datetime(stock_df['date'])

    # Create cross-reference chart
    st.markdown("### 📊 Multi-Topic Sentiment Cross-Reference")

    fig_cross = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            'Multiple Topics Sentiment Comparison',
            f'{stock_symbol.upper()} Stock Price'
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        vertical_spacing=0.15
    )

    topic_colors = ['#00bfff', '#00ff88', '#ff6b35', '#9d4edd', '#ffd700', '#39ff14']

    # Add each topic's sentiment
    for idx, (query_id, data) in enumerate(cross_ref_data.items()):
        color = topic_colors[idx % len(topic_colors)]
        fig_cross.add_trace(
            go.Scatter(
                x=data['daily']['date'],
                y=data['daily']['avg_sentiment'],
                name=data['query_text'],
                line=dict(color=color, width=3),
                mode='lines+markers',
                marker=dict(size=8)
            ),
            row=1, col=1
        )

    # Add stock price
    fig_cross.add_trace(
        go.Scatter(
            x=stock_df['date'],
            y=stock_df['Close'],
            name=f'{stock_symbol.upper()} Price',
            line=dict(color='#ff3366', width=4),
            mode='lines+markers',
            marker=dict(size=8),
            fill='tonexty',
            fillcolor='rgba(255, 51, 102, 0.1)'
        ),
        row=2, col=1
    )

    fig_cross.update_layout(
        height=700,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#0e1117',
        font=dict(color='#e0e0e0', size=12),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12)
        )
    )

    fig_cross.update_xaxes(showgrid=True, gridcolor='#3d3d3d')
    fig_cross.update_yaxes(showgrid=True, gridcolor='#3d3d3d')
    fig_cross.update_yaxes(title_text="Sentiment Score", row=1, col=1, title_font=dict(size=14))
    fig_cross.update_yaxes(title_text="Stock Price ($)", row=2, col=1, title_font=dict(size=14))

    st.plotly_chart(fig_cross, use_container_width=True)

    # Pattern Analysis
    st.markdown("### 🔍 Sentiment Pattern Analysis")

    # Merge all sentiments together
    merged_sentiments = stock_df[['date', 'Close', 'Daily_Return']].copy()

    for query_id, data in cross_ref_data.items():
        daily_data = data['daily'][['date', 'avg_sentiment']].copy()
        daily_data = daily_data.rename(columns={'avg_sentiment': f"sentiment_{data['query_text']}"})
        merged_sentiments = pd.merge(merged_sentiments, daily_data, on='date', how='outer')

    merged_sentiments = merged_sentiments.sort_values('date')
    merged_sentiments = merged_sentiments.dropna()

    if len(merged_sentiments) < 3:
        st.warning("Not enough overlapping data for pattern analysis")
    else:
        # Identify sentiment patterns
        sentiment_cols = [col for col in merged_sentiments.columns if col.startswith('sentiment_')]

        # Create pattern categories
        def categorize_pattern(row):
            sentiments = [row[col] for col in sentiment_cols]
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)

            if positive_count == len(sentiments):
                return "All Positive"
            elif negative_count == len(sentiments):
                return "All Negative"
            elif positive_count > negative_count:
                return "Mostly Positive"
            elif negative_count > positive_count:
                return "Mostly Negative"
            else:
                return "Mixed/Neutral"

        merged_sentiments['pattern'] = merged_sentiments.apply(categorize_pattern, axis=1)

        # Calculate average stock movement for each pattern
        pattern_analysis = merged_sentiments.groupby('pattern').agg({
            'Daily_Return': ['mean', 'std', 'count'],
            'Close': 'mean'
        }).round(3)

        pattern_analysis.columns = ['Avg Daily Return (%)', 'Std Dev', 'Days', 'Avg Stock Price']
        pattern_analysis = pattern_analysis.sort_values('Avg Daily Return (%)', ascending=False)

        st.markdown("#### 📈 Stock Performance by Sentiment Pattern")

        # Display metrics
        pattern_cols = st.columns(len(pattern_analysis))
        for idx, (pattern, row) in enumerate(pattern_analysis.iterrows()):
            with pattern_cols[idx]:
                emoji = "🟢" if row['Avg Daily Return (%)'] > 0 else "🔴" if row['Avg Daily Return (%)'] < 0 else "⚪"
                st.metric(
                    f"{emoji} {pattern}",
                    f"{row['Avg Daily Return (%)']:+.2f}%",
                    delta=f"{int(row['Days'])} days"
                )

        st.dataframe(
            pattern_analysis,
            use_container_width=True,
            height=200
        )

        # Detailed breakdown
        st.markdown("#### 📋 Pattern Breakdown by Date")

        pattern_display = merged_sentiments[['date', 'pattern', 'Daily_Return', 'Close'] + sentiment_cols].copy()
        pattern_display = pattern_display.sort_values('date', ascending=False)

        # Color code patterns
        def highlight_pattern(row):
            if row['pattern'] == 'All Positive':
                return ['background-color: #1a3d1a'] * len(row)
            elif row['pattern'] == 'All Negative':
                return ['background-color: #3d1a1a'] * len(row)
            elif row['pattern'] == 'Mostly Positive':
                return ['background-color: #1a2d1a'] * len(row)
            elif row['pattern'] == 'Mostly Negative':
                return ['background-color: #2d1a1a'] * len(row)
            else:
                return ['background-color: #1a1a1a'] * len(row)

        st.dataframe(
            pattern_display.head(50),
            use_container_width=True,
            height=300
        )

        # Insights
        best_pattern = pattern_analysis.iloc[0]
        worst_pattern = pattern_analysis.iloc[-1]

        st.markdown("### 💡 Cross-Reference Insights")
        st.info(f"""
**Best Pattern:** {best_pattern.name}
- Average daily return: **{best_pattern['Avg Daily Return (%)']:+.2f}%**
- Occurred on {int(best_pattern['Days'])} days

**Worst Pattern:** {worst_pattern.name}
- Average daily return: **{worst_pattern['Avg Daily Return (%)']:+.2f}%**
- Occurred on {int(worst_pattern['Days'])} days

**Topics Analyzed:** {', '.join([data['query_text'] for data in cross_ref_data.values()])}
        """)

    st.stop()

# Continue with single topic analysis
selected_query_id = selected_query_id

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

# Source Impact Analysis
st.markdown("---")
st.subheader("📡 Source Impact Analysis")
st.markdown(f"Compare how different news sources cover **'{selected_query['query_text']}'** and their impact on **{stock_symbol.upper()}**")

# Get unique sources
available_sources = df_articles['source'].value_counts()
source_options = list(available_sources.index)

if len(source_options) > 1:
    # Source selector
    col_src1, col_src2 = st.columns([3, 1])

    with col_src1:
        selected_sources = st.multiselect(
            "Select Sources to Compare",
            options=source_options,
            default=source_options[:min(5, len(source_options))],  # Default to first 5
            help="Toggle sources to compare their sentiment and impact"
        )

    with col_src2:
        if st.button("Select All Sources"):
            selected_sources = source_options
            st.rerun()

    if selected_sources:
        # Calculate per-source daily sentiment
        source_sentiment_data = []
        source_colors = ['#00bfff', '#00ff88', '#ff6b35', '#9d4edd', '#ffd700', '#ff3366', '#39ff14', '#ff00ff']

        # Create interactive chart comparing sources
        fig_sources = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f'Sentiment by Source: "{selected_query["query_text"]}"',
                f'{stock_symbol.upper()} Stock Price'
            ),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
            vertical_spacing=0.15
        )

        # Add line for each source
        for idx, source in enumerate(selected_sources):
            source_df = df_articles[df_articles['source'] == source].copy()
            source_daily = source_df.groupby('date').agg({
                'sentiment_polarity': 'mean',
                'title': 'count'
            }).reset_index()
            source_daily.columns = ['date', 'avg_sentiment', 'article_count']
            source_daily['date'] = pd.to_datetime(source_daily['date'])

            # Calculate correlation with stock returns
            source_merged = pd.merge(source_daily, stock_df, on='date', how='inner')
            if len(source_merged) > 2:
                source_corr = source_merged['avg_sentiment'].corr(source_merged['Daily_Return'])
            else:
                source_corr = 0.0

            source_sentiment_data.append({
                'source': source,
                'articles': len(source_df),
                'avg_sentiment': source_df['sentiment_polarity'].mean(),
                'correlation': source_corr,
                'days_covered': len(source_daily)
            })

            # Add to chart
            color = source_colors[idx % len(source_colors)]
            fig_sources.add_trace(
                go.Scatter(
                    x=source_daily['date'],
                    y=source_daily['avg_sentiment'],
                    name=f'{source} (r={source_corr:+.2f})',
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    marker=dict(size=6),
                    hovertemplate=f'<b>{source}</b><br>Date: %{{x}}<br>Sentiment: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add stock price to bottom chart
        fig_sources.add_trace(
            go.Scatter(
                x=merged['date'],
                y=merged['Close'],
                name=f'{stock_symbol.upper()} Price',
                line=dict(color='#ff3366', width=4),
                mode='lines+markers',
                marker=dict(size=8),
                fill='tonexty',
                fillcolor='rgba(255, 51, 102, 0.1)'
            ),
            row=2, col=1
        )

        # Layout
        fig_sources.update_layout(
            height=700,
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0e1117',
            font=dict(color='#e0e0e0', size=12),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=11)
            )
        )

        fig_sources.update_xaxes(showgrid=True, gridcolor='#3d3d3d')
        fig_sources.update_yaxes(showgrid=True, gridcolor='#3d3d3d')
        fig_sources.update_yaxes(title_text="Sentiment Score", row=1, col=1, title_font=dict(size=14))
        fig_sources.update_yaxes(title_text="Stock Price ($)", row=2, col=1, title_font=dict(size=14))

        st.plotly_chart(fig_sources, use_container_width=True)

        # Source comparison table
        st.markdown("#### 📊 Source Impact Comparison")
        source_comparison_df = pd.DataFrame(source_sentiment_data)
        source_comparison_df = source_comparison_df.sort_values('correlation', ascending=False, key=abs)

        # Display metrics
        cols = st.columns(len(selected_sources))
        for idx, row in source_comparison_df.iterrows():
            with cols[source_comparison_df.index.get_loc(idx)]:
                corr_emoji = "🔥" if abs(row['correlation']) > 0.5 else "📊" if abs(row['correlation']) > 0.3 else "📉"
                st.metric(
                    f"{corr_emoji} {row['source']}",
                    f"{row['correlation']:+.3f}",
                    delta=f"{row['articles']} articles"
                )

        # Detailed table
        st.dataframe(
            source_comparison_df,
            column_config={
                "source": "Source",
                "articles": "Articles",
                "avg_sentiment": st.column_config.NumberColumn("Avg Sentiment", format="%.3f"),
                "correlation": st.column_config.NumberColumn("Correlation", format="%.3f"),
                "days_covered": "Days"
            },
            use_container_width=True,
            height=200
        )

        # Impact interpretation
        st.markdown("#### 💡 Source Impact Insights")
        top_source = source_comparison_df.iloc[0]
        impact_emoji = "🔥" if abs(top_source['correlation']) > 0.5 else "📊"

        st.info(f"""
{impact_emoji} **Most Impactful Source: {top_source['source']}**

- **Correlation:** {top_source['correlation']:+.3f} with {stock_symbol.upper()} returns
- **Coverage:** {top_source['articles']} articles over {top_source['days_covered']} days
- **Avg Sentiment:** {top_source['avg_sentiment']:+.3f}

This source's sentiment about **"{selected_query['query_text']}"** shows the {'strongest' if abs(top_source['correlation']) > 0.3 else 'some'} correlation with stock movements.
        """)

        # Filter articles by source
        st.markdown("#### 🔍 Filter Articles by Source")
        source_filter = st.selectbox(
            "View articles from:",
            options=["All Sources"] + selected_sources,
            key="source_filter_select"
        )

        if source_filter != "All Sources":
            df_articles = df_articles[df_articles['source'] == source_filter]
            st.caption(f"Showing {len(df_articles)} articles from **{source_filter}**")
    else:
        st.warning("👆 Select at least one source to compare")
else:
    st.info("Only one source found in this dataset. Fetch more news to compare sources.")

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
