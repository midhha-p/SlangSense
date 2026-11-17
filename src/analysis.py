import re
import html

from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
    TfidfVectorizer
)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sentence_transformers import SentenceTransformer

from src.data_utils import clean_text, detect_slang


# --------------------------------------------------
# 1. SLANG FREQUENCY ANALYSIS
# --------------------------------------------------

def analyse_slang_frequency(
    tweets_df,
    slang_dictionary,
    max_tweets=None
):

    if "tweet" not in tweets_df.columns:
        raise ValueError(
            "The tweets dataset must contain a 'tweet' column."
        )

    tweets = tweets_df["tweet"].dropna()

    if max_tweets is not None:
        tweets = tweets.head(max_tweets)

    terms = sorted(
        slang_dictionary.keys(),
        key=len,
        reverse=True
    )

    if not terms:
        return pd.DataFrame(
            columns=["slang_term", "tweet_count"]
        )

    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(term) for term in terms)
        + r")(?!\w)",
        flags=re.IGNORECASE
    )

    counts = Counter()

    for tweet in tweets:

        cleaned = clean_text(tweet)

        found_terms = set(
            match.group(0).lower()
            for match in pattern.finditer(cleaned)
        )

        counts.update(found_terms)

    result = pd.DataFrame([
        {
            "slang_term": term,
            "tweet_count": count
        }
        for term, count in counts.items()
    ])

    if result.empty:
        return pd.DataFrame(
            columns=["slang_term", "tweet_count"]
        )

    return (
        result.sort_values(
            "tweet_count",
            ascending=False
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------
# 2. SLANG DETECTION
# --------------------------------------------------

def analyse_tweet_slang(text, slang_dictionary):

    return detect_slang(text, slang_dictionary)


# --------------------------------------------------
# 3. ENGAGEMENT ANALYSIS
# --------------------------------------------------

def analyse_engagement(trends_df):

    metrics = [
        "Views",
        "Likes",
        "Shares",
        "Comments"
    ]

    available_metrics = [
        column
        for column in metrics
        if column in trends_df.columns
    ]

    if not available_metrics:
        return pd.DataFrame()

    numeric_data = trends_df[available_metrics].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return (
        numeric_data
        .describe()
        .T
        .reset_index(names="metric")
    )


# --------------------------------------------------
# 4. ENGAGEMENT BY PLATFORM
# --------------------------------------------------

def engagement_by_platform(trends_df):

    required = [
        "Platform",
        "Views",
        "Likes",
        "Shares",
        "Comments"
    ]

    if not all(
        column in trends_df.columns
        for column in required
    ):
        return pd.DataFrame()

    df = trends_df.copy()

    for column in required[1:]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    result = (
        df.groupby("Platform")[required[1:]]
        .mean()
        .round(2)
        .reset_index()
        .sort_values(
            "Views",
            ascending=False
        )
    )

    return result


# --------------------------------------------------
# 5. SEMANTIC MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_semantic_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 6. TWEET PREPROCESSING AND TOPIC ANALYSIS
# SENTENCE-BERT + K-MEANS
# --------------------------------------------------

def discover_topics(
    tweets_df,
    number_of_topics=5,
    max_tweets=3000
):

    # ---------------------------------
    # 1. Validate input
    # ---------------------------------

    if tweets_df is None or tweets_df.empty:
        raise ValueError(
            "No tweets available for topic analysis."
        )

    if "tweet" not in tweets_df.columns:
        raise ValueError(
            "The dataset must contain a 'tweet' column."
        )

    if number_of_topics < 2:
        raise ValueError(
            "At least 2 topics are required."
        )

    if max_tweets < 1:
        raise ValueError(
            "max_tweets must be greater than zero."
        )

    # ---------------------------------
    # 2. Custom stopwords
    # ---------------------------------

    custom_stopwords = set(ENGLISH_STOP_WORDS)

    custom_stopwords.update({

        "just", "like", "know", "got", "going",
        "really", "don", "ll", "ve", "im",
        "ive", "dont", "didnt", "doesnt",
        "thats", "youre", "theyre", "wanna",
        "gonna", "yeah", "yes", "lol", "haha",
        "hahaha", "guys", "thanks", "thank",
        "hey", "oh", "amp", "rt", "http",
        "https", "com", "www"

    })

    stopword_list = sorted(custom_stopwords)

    # ---------------------------------
    # 3. Clean tweets
    # ---------------------------------

    def clean_tweet(text):

        text = html.unescape(str(text))
        text = text.lower()

        text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            text
        )

        text = re.sub(
            r"@\w+",
            " ",
            text
        )

        text = re.sub(
            r"#",
            "",
            text
        )

        text = re.sub(
            r"\brt\b",
            " ",
            text
        )

        text = re.sub(
            r"[^a-z\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    tweets = (
        tweets_df["tweet"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(max_tweets)
    )

    cleaned_tweets = tweets.apply(clean_tweet)

    valid_mask = (
        cleaned_tweets.str.split().str.len() >= 3
    )

    cleaned_tweets = cleaned_tweets[valid_mask]

    original_tweets = tweets.loc[
        cleaned_tweets.index
    ]

    if len(cleaned_tweets) < number_of_topics * 2:

        raise ValueError(
            "Not enough valid tweets for the selected number of topics."
        )

    documents = cleaned_tweets.tolist()

    # ---------------------------------
    # 4. Load Sentence-BERT
    # ---------------------------------

    semantic_model = get_semantic_model()

    # ---------------------------------
    # 5. Generate semantic embeddings
    # ---------------------------------

    embeddings = semantic_model.encode(
        documents,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # ---------------------------------
    # 6. K-Means clustering
    # ---------------------------------

    actual_topics = min(
        number_of_topics,
        len(documents) - 1
    )

    model = KMeans(
        n_clusters=actual_topics,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(
        embeddings
    )

    # ---------------------------------
    # 7. Silhouette score
    # ---------------------------------

    if (
        1 < len(set(cluster_labels)) < len(documents)
    ):

        overall_silhouette = float(
            silhouette_score(
                embeddings,
                cluster_labels,
                metric="cosine"
            )
        )

    else:

        overall_silhouette = 0.0

    # ---------------------------------
    # 8. Improved keyword extraction
    # ---------------------------------

    tfidf_vectorizer = TfidfVectorizer(
        stop_words=stopword_list,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True
    )

    try:

        tfidf_matrix = tfidf_vectorizer.fit_transform(
            documents
        )

    except ValueError as error:

        raise ValueError(
            "Unable to extract topic keywords. "
            "Try analysing more tweets."
        ) from error

    feature_names = (
        tfidf_vectorizer.get_feature_names_out()
    )

    # ---------------------------------
    # 9. Binary matrix for UMass coherence
    # ---------------------------------

    count_vectorizer = CountVectorizer(
        stop_words=stopword_list,
        binary=True,
        max_features=5000,
        min_df=2
    )

    binary_matrix = count_vectorizer.fit_transform(
        documents
    )

    count_features = (
        count_vectorizer.get_feature_names_out()
    )

    binary_lookup = {
        word: index
        for index, word in enumerate(count_features)
    }

    # ---------------------------------
    # 10. Prepare topic results
    # ---------------------------------

    results = []

    for topic_id in sorted(set(cluster_labels)):

        indices = np.where(
            cluster_labels == topic_id
        )[0]

        topic_embeddings = embeddings[indices]

        topic_tfidf = tfidf_matrix[indices]

        cluster_size = len(indices)

        # ---------------------------------
        # 11. Extract topic keywords
        # ---------------------------------

        mean_scores = np.asarray(
            topic_tfidf.mean(axis=0)
        ).ravel()

        top_indices = mean_scores.argsort()[::-1]

        keywords = [
            feature_names[index]
            for index in top_indices
            if mean_scores[index] > 0
        ][:10]

        # ---------------------------------
        # 12. Generate topic label
        # ---------------------------------

        topic_label = (
            ", ".join(
                word.title()
                for word in keywords[:3]
            )
            if keywords
            else "Unidentified Topic"
        )

        # ---------------------------------
        # 13. Semantic cohesion
        # ---------------------------------

        centroid = topic_embeddings.mean(axis=0)

        centroid_norm = np.linalg.norm(centroid)

        if centroid_norm > 0:

            centroid = centroid / centroid_norm

        similarities = topic_embeddings @ centroid

        semantic_cohesion = float(
            np.mean(similarities)
        )

        # ---------------------------------
        # 14. Keyword relevance
        # ---------------------------------

        if keywords:

            keyword_text = (
                "This topic discusses "
                + ", ".join(keywords)
            )

            keyword_embedding = semantic_model.encode(
                [keyword_text],
                normalize_embeddings=True,
                convert_to_numpy=True
            )[0]

            keyword_relevance = float(
                np.mean(
                    topic_embeddings @ keyword_embedding
                )
            )

        else:

            keyword_relevance = 0.0

        # ---------------------------------
        # 15. UMass coherence
        # ---------------------------------

        coherence_values = []

        valid_keywords = [
            word
            for word in keywords
            if word in binary_lookup
        ]

        for i in range(1, len(valid_keywords)):

            for j in range(i):

                word_i = binary_lookup[
                    valid_keywords[i]
                ]

                word_j = binary_lookup[
                    valid_keywords[j]
                ]

                documents_i = binary_matrix[:, word_i]

                documents_j = binary_matrix[:, word_j]

                co_occurrence = (
                    documents_i.multiply(documents_j)
                ).sum()

                frequency_j = int(
                    documents_j.sum()
                )

                score = np.log(
                    (co_occurrence + 1) /
                    max(frequency_j, 1)
                )

                coherence_values.append(
                    float(score)
                )

        topic_coherence = (
            float(np.mean(coherence_values))
            if coherence_values
            else 0.0
        )

        # ---------------------------------
        # 16. Representative tweet selection
        # ---------------------------------

        representative_order = np.argsort(
            similarities
        )[::-1]

        selected_positions = representative_order[:3]

        # ---------------------------------
        # 17. Store topic results
        # ---------------------------------

        for position in selected_positions:

            selected_index = indices[position]

            representative_similarity = float(
                similarities[position]
            )

            validation_status = (
                "Review"
                if representative_similarity < 0.30
                else "High similarity"
            )

            results.append({

                "topic": int(topic_id) + 1,

                "topic_label": topic_label,

                "tweet": original_tweets.iloc[
                    selected_index
                ],

                "keywords": keywords,

                "cluster_size": cluster_size,

                "silhouette_score": float(
                    overall_silhouette
                ),

                "topic_weight": representative_similarity,

                "representative_similarity": (
                    representative_similarity
                ),

                "validation_status": validation_status,

                "topic_coherence": topic_coherence,

                "topic_similarity": semantic_cohesion,

                "keyword_relevance": keyword_relevance,

                "topic_keyword_similarity": keyword_relevance

            })

    return pd.DataFrame(results)
