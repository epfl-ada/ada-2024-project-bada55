import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time
import os
import nltk
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from src.visualization.sentiment_analysis_viz import *

def analyze_and_classify_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
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

    nltk.download('vader_lexicon')
    print("downloaded")

    # BeerAdvocate
    sentiment_results_ba = ba_reviews_sentiment["text"].apply(analyze_and_classify_sentiment)
    ba_reviews_sentiment["sentiment_scores"] = sentiment_results_ba.apply(lambda x: x[0])  
    ba_reviews_sentiment["sentiment_label"] = sentiment_results_ba.apply(lambda x: x[1])
    ba_reviews_sentiment["sentiment_scores"] = ba_reviews_sentiment["sentiment_scores"].astype(str)

    ba_reviews_sentiment["positive_sentiment"] = ba_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['pos'])
    ba_reviews_sentiment["negative_sentiment"] = ba_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['neg'])
    ba_reviews_sentiment["neutral_sentiment"] = ba_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['neu'])

    ba_reviews_sentiment["compound_sentiment"] = ba_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['compound'])

    print("fini BeerAdvocate")

    # RateBeer
    sentiment_results_rb = rb_reviews_sentiment["text"].apply(analyze_and_classify_sentiment)
    rb_reviews_sentiment["sentiment_scores"] = sentiment_results_rb.apply(lambda x: x[0])  
    rb_reviews_sentiment["sentiment_label"] = sentiment_results_rb.apply(lambda x: x[1])
    rb_reviews_sentiment["sentiment_scores"] = rb_reviews_sentiment["sentiment_scores"].astype(str)

    rb_reviews_sentiment["positive_sentiment"] = rb_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['pos'])
    rb_reviews_sentiment["negative_sentiment"] = rb_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['neg'])
    rb_reviews_sentiment["neutral_sentiment"] = rb_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['neu'])

    rb_reviews_sentiment["compound_sentiment"] = rb_reviews_sentiment["sentiment_scores"].apply(eval).apply(lambda x: x['compound'])

    return (
            fig_distribution_sentiment(ba_reviews_sentiment),
            fig_compound_sentiment(ba_reviews_sentiment),
            fig_percentage(ba_reviews_sentiment),
            fig_distribution_sentiment(rb_reviews_sentiment),
            fig_compound_sentiment(rb_reviews_sentiment),
            fig_percentage(rb_reviews_sentiment)
    )
