import pandas as pd
import plotly.express as px
from src.visualization.sentiment_analysis_viz import *
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyze_and_classify_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        sentiment_label = "positive"
    elif compound_score <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    return scores, sentiment_label

def sentiment_analysis_analyser(ba_reviews: pd.DataFrame, rb_reviews: pd.DataFrame) -> pd.DataFrame:

    ba_reviews_sentiment = ba_reviews.copy()
    rb_reviews_sentiment = rb_reviews.copy()

    # BeerAdvocate
    sentiment_results_ba = ba_reviews_sentiment["text"].apply(lambda text: analyze_and_classify_sentiment(text))
    ba_reviews_sentiment["sentiment_scores"] = sentiment_results_ba.apply(lambda x: x[0])  
    ba_reviews_sentiment["sentiment_label"] = sentiment_results_ba.apply(lambda x: x[1])
    ba_reviews_sentiment["sentiment_scores"] = ba_reviews_sentiment["sentiment_scores"].astype(str)

    ba_reviews_sentiment["positive_sentiment"] = ba_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['pos'])
    ba_reviews_sentiment["negative_sentiment"] = ba_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['neg'])
    ba_reviews_sentiment["neutral_sentiment"] = ba_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['neu'])

    ba_reviews_sentiment["compound_sentiment"] = ba_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['compound'])

    # RateBeer
    sentiment_results_rb = rb_reviews_sentiment["text"].apply(lambda text: analyze_and_classify_sentiment(text))
    rb_reviews_sentiment["sentiment_scores"] = sentiment_results_rb.apply(lambda x: x[0])  
    rb_reviews_sentiment["sentiment_label"] = sentiment_results_rb.apply(lambda x: x[1])
    rb_reviews_sentiment["sentiment_scores"] = rb_reviews_sentiment["sentiment_scores"].astype(str)

    rb_reviews_sentiment["positive_sentiment"] = rb_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['pos'])
    rb_reviews_sentiment["negative_sentiment"] = rb_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['neg'])
    rb_reviews_sentiment["neutral_sentiment"] = rb_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['neu'])

    rb_reviews_sentiment["compound_sentiment"] = rb_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['compound'])

    # Comparaison
    def calculate_sentiment_distribution(df):
        sentiment_counts = df["sentiment_label"].value_counts(normalize=True) * 100
        return {
            "positive": sentiment_counts.get("positive", 0),
            "negative": sentiment_counts.get("negative", 0),
            "neutral": sentiment_counts.get("neutral", 0),
        }

    beeradvocate_distribution = calculate_sentiment_distribution(ba_reviews_sentiment)
    ratebeer_distribution = calculate_sentiment_distribution(rb_reviews_sentiment)

    ratebeer_sizes = [ratebeer_distribution["positive"], ratebeer_distribution["negative"], ratebeer_distribution["neutral"]]
    beeradvocate_sizes = [beeradvocate_distribution["positive"], beeradvocate_distribution["negative"], beeradvocate_distribution["neutral"]]
    labels = ["Positive", "Negative", "Neutral"]

    fig_list = [
            fig_distribution_sentiment(ba_reviews_sentiment, title="BeerAdvocate"),
            fig_compound_sentiment(ba_reviews_sentiment, title="BeerAdvocate"),
            fig_percentage(ba_reviews_sentiment, title="BeerAdvocate"),
            fig_distribution_sentiment(rb_reviews_sentiment, title="RateBeer"),
            fig_compound_sentiment(rb_reviews_sentiment, title="RateBeer"),
            fig_percentage(rb_reviews_sentiment, title="RateBeer"),
            fig_sentiment_distribution(beeradvocate_sizes, ratebeer_sizes,  labels),
            fig_distribution_sentiment_combined(ba_reviews_sentiment, rb_reviews_sentiment, "BeerAdvocate", "RateBeer"),
    ]
    return fig_list
