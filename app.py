
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_utils import load_datasets, get_slang_dictionary
from src.analysis import (
    analyse_slang_frequency,
    analyse_tweet_slang,
    analyse_engagement,
    discover_topics,
    engagement_by_platform
)
from src.ml_model import train_model, predict_category


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SlangSense | Social Language Intelligence",
    page_icon="🗯️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# THEME AND STYLING
# =========================================================

st.html("""
<style>

/* Main background */
.stApp {
    background-color: #111827;
    color: #E5E7EB;
}

/* Main content spacing */
[data-testid="stMain"] .block-container {
    max-width: 1350px;
    padding: 2rem 2.5rem 3rem 2.5rem;
}

/* Main headings */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #F9FAFB !important;
    font-weight: 650 !important;
}

/* Main text */
[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label {
    color: #D1D5DB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #172033;
    border-right: 1px solid #293548;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F9FAFB !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #D1D5DB !important;
}
/* Sidebar navigation buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    color: #C7D1E2 !important;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 12px 15px;
    font-size: 14px;
    font-weight: 500;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #263650;
    border-color: #526B9E;
}

/* Active sidebar button */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #344968;
    border: 1px solid #7188B7;
    color: #FFFFFF !important;
}

/* Project header - fully rounded rectangle */
.hero {
    background: linear-gradient(135deg, #202D43, #1B2738);
    border: 1px solid #52658A;
    border-radius: 24px;
    padding: 65px 45px;
    margin: 45px 0 40px 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.22);
    text-align: left;
}

.hero-label {
    color: #A5B4D9 !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 14px;
}

.hero-title {
    color: #FFFFFF !important;
    font-size: 42px;
    font-weight: 750;
    letter-spacing: -0.8px;
    margin: 0;
}

.hero-subtitle {
    color: #C2CCDD !important;
    font-size: 15px;
    margin-top: 12px;
    line-height: 1.7;
}

/* Section labels */
.section-label {
    color: #A5B4D9 !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
    margin-bottom: 5px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #1B2738;
    border: 1px solid #3B4B68;
    border-radius: 15px;
    padding: 18px 20px;
    min-height: 105px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.12);
}

[data-testid="stMetricLabel"] p {
    color: #AEBBD0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Text inputs */
.stTextInput input,
.stTextArea textarea {
    background: #1B2738 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #465675 !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #8998B1 !important;
    -webkit-text-fill-color: #8998B1 !important;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background: #1B2738 !important;
    color: #FFFFFF !important;
    border-color: #465675 !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] span {
    color: #FFFFFF !important;
}

/* Dropdown */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #1B2738 !important;
}

[data-baseweb="menu"] li {
    color: #FFFFFF !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #1B2738;
    border: 1px solid #3B4B68;
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
    font-weight: 600;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: #526B9E;
    color: #FFFFFF !important;
    border: 1px solid #7188B7;
    border-radius: 10px;
    padding: 9px 20px;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #657EAE;
    border-color: #A5B4D9;
    color: #FFFFFF !important;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #3B4B68;
    border-radius: 12px;
    overflow: hidden;
}

/* Captions */
[data-testid="stCaptionContainer"] p {
    color: #AAB6C6 !important;
}

/* Dividers */
hr {
    border-color: #344256 !important;
}

/* Hide footer */
footer {
    visibility: hidden;
}

</style>
""")


# =========================================================
# CHART SETTINGS
# =========================================================

CHART_BG = "#111827"
PLOT_BG = "#1B2738"
TEXT_COLOR = "#D1D5DB"

px.defaults.template = "plotly_dark"

px.defaults.color_discrete_sequence = [
    "#8195C8",
    "#60A5FA",
    "#A1A1AA",
    "#C4B5FD",
    "#94A3B8"
]


def style_chart(fig, height=420):

    fig.update_layout(
        height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(
            family="Arial",
            size=13,
            color=TEXT_COLOR
        ),
        title_font=dict(
            size=18,
            color="#F9FAFB"
        ),
        margin=dict(
            l=45,
            r=35,
            t=60,
            b=45
        ),
        legend=dict(
            font=dict(color=TEXT_COLOR),
            title=dict(text="")
        )
    )

    fig.update_xaxes(
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        gridcolor="#344256",
        zerolinecolor="#344256"
    )

    fig.update_yaxes(
        title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        gridcolor="#344256",
        zerolinecolor="#344256",
        tickformat=","
    )

    return fig


# =========================================================
# PROJECT HEADER
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-label">
        SOCIAL LANGUAGE INTELLIGENCE PLATFORM
    </div>

    <div class="hero-title">
        🗯️ SlangSense
    </div>

    <div class="hero-subtitle">
        Gen-Z Slang Detection &amp; Emerging Trend Analysis
    </div>

</div>
""")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div style="padding: 8px 0 15px 0;">
    <div style="font-size:25px; font-weight:750; color:#FFFFFF;">
        🗯️ SlangSense
    </div>
    <div style="font-size:10px; letter-spacing:2px; color:#A5B4D9;">
        SOCIAL LANGUAGE ANALYTICS
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="margin-bottom:8px;">
    <div style="
        color:#FFFFFF;
        font-size:15px;
        font-weight:750;
        margin-bottom:7px;">
        About SlangSense
    </div>

  <div style="
    color:#C2CCDD;
    font-size:13px;
    font-weight:400;
    line-height:1.7;">
    Explore Gen-Z slang, discover social conversation topics
    and understand emerging online trends.
</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "<div style='font-size:11px; font-weight:700; color:#A5B4D9; letter-spacing:1.5px;'>EXPLORE</div>",
    unsafe_allow_html=True
)

pages = [
    "Slang Detector",
    "Slang Frequency",
    "Topic Analysis",
    "Engagement Analysis"
]

if "page" not in st.session_state:
    st.session_state["page"] = "Slang Detector"

for item in pages:

    symbol = "●" if st.session_state["page"] == item else "○"

    if st.sidebar.button(
        f"{symbol}　{item}",
        key=f"nav_{item}",
        use_container_width=True,
        type="primary" if st.session_state["page"] == item else "secondary"
    ):
        st.session_state["page"] = item
        st.rerun()

page = st.session_state["page"]

st.sidebar.markdown("---")




# =========================================================
# LOAD DATASETS
# =========================================================

@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    return load_datasets()


try:

    slang_df, tweets_df, trends_df = load_data()
    slang_dictionary = get_slang_dictionary(slang_df)

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()


# =========================================================
# 1. SLANG DETECTOR AND ML PREDICTION
# =========================================================

if page == "Slang Detector":

    st.markdown(
        '<div class="section-label">TEXT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.header("Slang Detector")

    st.caption(
        "Identify Gen-Z slang and predict its language category."
    )

    text_input = st.text_area(
        "Enter social media text",
        placeholder="Example: That outfit ate, the vibe is fire!",
        height=130
    )

    if st.button("Detect Slang", key="detect_slang"):

        if not text_input.strip():

            st.warning("Please enter some text.")

        else:

            results = analyse_tweet_slang(
                text_input,
                slang_dictionary
            )

            if results:

                st.success(
                    f"{len(results)} recognised slang term(s) found."
                )

                result_df = pd.DataFrame(results)

                result_df.columns = [
                    "Slang Term",
                    "Meaning"
                ]

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No matching slang terms were found.")

    st.divider()

    st.markdown(
        '<div class="section-label">MACHINE LEARNING</div>',
        unsafe_allow_html=True
    )

    st.subheader("Slang Category Prediction")

    st.caption(
        "TF-IDF feature extraction with a Linear Support Vector Machine."
    )

    if st.button("Train Category Model", key="train_category_model"):

        try:

            with st.spinner("Training the category model..."):

                (
                    model,
                    accuracy,
                    report_df,
                    matrix_df,
                    train_size,
                    test_size,
                    macro_f1,
                    weighted_f1
                ) = train_model()

            st.session_state["slang_ml_model"] = model
            st.session_state["slang_ml_accuracy"] = accuracy

            st.success("Category model trained successfully.")

        except Exception as error:

            st.error(f"Model training failed: {error}")

    if "slang_ml_model" in st.session_state:

        accuracy = st.session_state.get(
            "slang_ml_accuracy",
            None
        )

        if accuracy is not None:

            st.metric(
                "Model Test Accuracy",
                f"{accuracy:.2%}"
            )

        prediction_text = st.text_input(
            "Enter a sentence for category prediction",
            placeholder="Example: That movie was mid.",
            key="prediction_text"
        )

        if st.button("Predict Category", key="predict_category"):

            if prediction_text.strip():

                try:

                    prediction = predict_category(
                        st.session_state["slang_ml_model"],
                        prediction_text
                    )

                    st.success(
                        f"Predicted Category: {str(prediction).title()}"
                    )

                except Exception as error:

                    st.error(f"Prediction failed: {error}")

            else:

                st.warning("Please enter a sentence.")

    else:

        st.info(
            "Train the category model to enable sentence prediction."
        )


# =========================================================
# 2. SLANG FREQUENCY
# =========================================================

elif page == "Slang Frequency":

    st.markdown(
        '<div class="section-label">TREND ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.header("Slang Frequency Analysis")

    st.caption(
        "Explore the frequency of recognised slang terms in the tweet dataset."
    )

    # -----------------------------------------------------
    # SAMPLE SIZE
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [2, 1],
        vertical_alignment="bottom"
    )

    with col1:
        sample_size = st.selectbox(
            "Tweet Sample Size",
            [1000, 5000, 10000, 25000, "All tweets"],
            index=1
        )

    with col2:
        analyse_button = st.button(
            "Analyse Frequency",
            key="analyse_frequency",
            use_container_width=True
        )

    # -----------------------------------------------------
    # ANALYSE FREQUENCY
    # -----------------------------------------------------

    if analyse_button:

        limit = (
            None if sample_size == "All tweets"
            else int(sample_size)
        )

        with st.spinner("Analysing slang frequency..."):

            frequency_df = analyse_slang_frequency(
                tweets_df,
                slang_dictionary,
                max_tweets=limit
            )

        st.session_state["frequency_results"] = frequency_df

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    if "frequency_results" in st.session_state:

        frequency_df = st.session_state["frequency_results"]

        if frequency_df.empty:

            st.warning("No recognised slang terms found.")

        else:

            frequency_df = (
                frequency_df
                .sort_values("tweet_count", ascending=False)
                .reset_index(drop=True)
            )

            st.session_state["frequency_results"] = frequency_df

            st.markdown("---")

            st.metric(
                "Recognised Slang Terms",
                f"{len(frequency_df):,}"
            )

            # =================================================
            # CHART 1 - TOP 10 SLANG TERMS
            # =================================================

            st.markdown("### Top 10 Most Frequently Used Slang")

            top_data = (
                frequency_df.head(10)
                .sort_values("tweet_count", ascending=False)
            )

            fig1 = px.bar(
                top_data,
                x="slang_term",
                y="tweet_count",
                text="tweet_count",
                labels={
                    "slang_term": "Slang Term",
                    "tweet_count": "Tweet Count"
                }
            )

            fig1.update_traces(
                marker_color="#8195C8",
                texttemplate="%{text:,}",
                textposition="outside",
                textfont=dict(size=11, color="#E5E7EB"),
                cliponaxis=False
            )

            fig1.update_layout(
                height=340,
                bargap=0.4,
                showlegend=False,
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#D1D5DB", size=12),
                margin=dict(l=15, r=25, t=25, b=55),
                xaxis=dict(
                    title="Slang Term",
                    showgrid=False,
                    tickangle=-30
                ),
                yaxis=dict(
                    title="Tweet Count",
                    showgrid=True,
                    gridcolor="#344256",
                    rangemode="tozero"
                )
            )

            st.plotly_chart(
                fig1,
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False}
            )

            st.markdown("---")

            # =================================================
            # CHART 2 - COMPLETE FREQUENCY DISTRIBUTION
            # =================================================

            st.markdown("### Complete Slang Frequency Distribution")

            all_data = frequency_df.sort_values(
                "tweet_count",
                ascending=False
            )

            fig2 = px.bar(
                all_data,
                x="slang_term",
                y="tweet_count",
                labels={
                    "slang_term": "Slang Term",
                    "tweet_count": "Tweet Count"
                }
            )

            fig2.update_traces(
                marker_color="#526B9E",
                marker_line_width=0,
                opacity=0.9
            )

            fig2.update_layout(
                height=360,
                bargap=0.25,
                showlegend=False,
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#D1D5DB", size=11),
                margin=dict(l=15, r=15, t=20, b=100),
                xaxis=dict(
                    title="Slang Terms",
                    showgrid=False,
                    tickangle=-60,
                    tickfont=dict(size=9)
                ),
                yaxis=dict(
                    title="Tweet Count",
                    showgrid=True,
                    gridcolor="#344256",
                    rangemode="tozero"
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False}
            )

            st.markdown("---")

            # =================================================
            # COMPLETE FREQUENCY TABLE
            # =================================================

            st.markdown("### Complete Frequency Results")

            st.dataframe(
                frequency_df,
                use_container_width=True,
                hide_index=True,
                height=350,
                column_config={
                    "slang_term": st.column_config.TextColumn(
                        "Slang Term"
                    ),
                    "tweet_count": st.column_config.NumberColumn(
                        "Tweet Count",
                        format="%d"
                    )
                }
            )

            # =================================================
            # DOWNLOAD RESULTS
            # =================================================

            st.download_button(
                "Download Results",
                frequency_df.to_csv(
                    index=False
                ).encode("utf-8"),
                "slang_frequency.csv",
                "text/csv"
            )

# =========================================================
# 3. TOPIC ANALYSIS
# =========================================================

elif page == "Topic Analysis":

    st.markdown(
        '<div class="section-label">CONVERSATION INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.header("Topic Analysis")

    st.caption(
        "Discover recurring themes and related words across social media conversations."
    )

    col1, col2 = st.columns(2)

    with col1:

        number_of_topics = st.slider(
            "Number of Topics",
            min_value=2,
            max_value=10,
            value=5
        )

    with col2:

        max_tweets = st.selectbox(
            "Tweets to Analyse",
            [500, 1000, 2000, 3000],
            index=1
        )

    if st.button("Discover Topics", key="discover_topics"):

        with st.spinner("Discovering conversation topics..."):

            try:

                topics_df = discover_topics(
                    tweets_df,
                    number_of_topics=number_of_topics,
                    max_tweets=max_tweets
                )

                st.session_state["topics_results"] = topics_df

            except Exception as error:

                st.error(f"Topic analysis failed: {error}")

    if "topics_results" in st.session_state:

        topics_df = st.session_state["topics_results"]

        if topics_df is None or topics_df.empty:

            st.warning("No topics could be discovered.")

        else:


            total_tweets = int(
                topics_df.groupby("topic")["cluster_size"]
                .first()
                .sum()
            )

            topic_count = topics_df["topic"].nunique()

            col1, col2 = st.columns(2)

            col1.metric(
                "Tweets Analysed",
                f"{total_tweets:,}"
            )

            col2.metric(
                "Topics Discovered",
                f"{topic_count:,}"
            )

            st.divider()

            topic_summary = (
                topics_df[
                    ["topic", "cluster_size"]
                ]
                .drop_duplicates(subset="topic")
                .sort_values("topic")
            )

            fig = px.bar(
                topic_summary,
                x="topic",
                y="cluster_size",
                text="cluster_size",
                color="cluster_size",
                color_continuous_scale=[
                    "#354D75",
                    "#526B9E",
                    "#8195C8"
                ],
                labels={
                    "topic": "Topic",
                    "cluster_size": "Number of Tweets"
                },
                title="Tweet Distribution Across Topics"
            )

            fig.update_traces(
                textposition="outside",
                textfont=dict(color="#F9FAFB")
            )

            fig.update_layout(
                coloraxis_showscale=False
            )

            st.plotly_chart(
                style_chart(fig, 380),
                use_container_width=True,
                theme=None
            )

            st.divider()

            st.subheader("Discovered Topics")

            for topic_number in sorted(
                topics_df["topic"].unique()
            ):

                topic_data = topics_df[
                    topics_df["topic"] == topic_number
                ]

                first_row = topic_data.iloc[0]

                cluster_size = int(
                    first_row["cluster_size"]
                )

                with st.expander(
                    f"Topic {topic_number} | {cluster_size:,} tweets",
                    expanded=False
                ):

                    if "topic_label" in topic_data.columns:

                        st.markdown(
                            f"**Topic Label:** {first_row['topic_label']}"
                        )

                    st.markdown("**Keywords**")
                    st.write(first_row["keywords"])

                    score_columns = []

                    if "topic_coherence" in topic_data.columns:
                        score_columns.append(
                            ("UMass Coherence", "topic_coherence")
                        )

                    if "topic_similarity" in topic_data.columns:
                        score_columns.append(
                            ("Semantic Cohesion", "topic_similarity")
                        )

                    if "topic_keyword_similarity" in topic_data.columns:
                        score_columns.append(
                            ("Keyword Relevance", "topic_keyword_similarity")
                        )

                    if "representative_similarity" in topic_data.columns:
                        score_columns.append(
                            ("Representative Similarity", "representative_similarity")
                        )

                    for start in range(0, len(score_columns), 2):

                        selected_scores = score_columns[
                            start:start + 2
                        ]

                        score_cols = st.columns(
                            len(selected_scores)
                        )

                        for score_col, (label, column) in zip(
                            score_cols,
                            selected_scores
                        ):

                            value = first_row[column]

                            if pd.notna(value):

                                score_col.metric(
                                    label,
                                    f"{float(value):.4f}"
                                )

                    if "validation_status" in topic_data.columns:

                        st.write(
                            "**Validation:**",
                            first_row["validation_status"]
                        )

                    st.markdown("**Representative Tweets**")

                    for tweet in topic_data["tweet"].head(3):

                        st.write(f"• {tweet}")

            st.download_button(
                "Download Topic Results",
                topics_df.to_csv(
                    index=False
                ).encode("utf-8"),
                "topic_analysis.csv",
                "text/csv"
            )


# =========================================================
# 4. ENGAGEMENT ANALYSIS
# =========================================================

elif page == "Engagement Analysis":

    st.markdown(
        '<div class="section-label">SOCIAL MEDIA PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    st.header("Engagement Analysis")

    st.caption(
        "Compare social media engagement across different platforms."
    )

    # -----------------------------------------------------
    # PREPARE ENGAGEMENT DATA
    # -----------------------------------------------------

    platform_df = engagement_by_platform(trends_df)

    if platform_df is not None and not platform_df.empty:

        available_metrics = [
            metric
            for metric in ["Views", "Likes", "Shares", "Comments"]
            if metric in platform_df.columns
        ]

        if available_metrics:

            # -------------------------------------------------
            # PLATFORM COMPARISON
            # -------------------------------------------------

            st.subheader("Platform Comparison")

            selected_metric = st.selectbox(
                "Choose Engagement Metric",
                available_metrics
            )

            fig = px.bar(
                platform_df,
                x="Platform",
                y=selected_metric,
                color="Platform",
                text=selected_metric,
                labels={
                    "Platform": "Social Media Platform",
                    selected_metric: f"Average {selected_metric}"
                },
                title=f"Average {selected_metric} by Platform",
                color_discrete_sequence=[
                    "#8195C8",
                    "#60A5FA",
                    "#A1A1AA",
                    "#C4B5FD"
                ]
            )

            fig.update_traces(
                texttemplate="%{text:,.0f}",
                textposition="outside",
                cliponaxis=False
            )

            st.plotly_chart(
                style_chart(fig, 440),
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False}
            )

        else:

            st.warning("No engagement metrics are available.")

    else:

        st.warning("Platform engagement data is not available.")

    