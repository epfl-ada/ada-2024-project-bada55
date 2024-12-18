import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def fig_distribution_sentiment(reviews: pd.DataFrame) -> plt.Figure:
    plt.figure(figsize=(10, 6))
    plt.hist(
        reviews["positive_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Positive Sentiment", color='green'
    )
    plt.hist(
        reviews["negative_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Negative Sentiment", color='salmon'
    )
    plt.hist(
        reviews["neutral_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Neutral Sentiment", color='blue'
    )
    plt.xlim([0, 1])
    plt.xlabel("Sentiment Score")
    plt.ylabel("Number of Reviews")
    plt.title("Distribution of Positive, Negative, and Neutral Sentiment in Reviews of BeerAdvocate")
    plt.legend()

    return plt.gcf()

def fig_compound_sentiment(reviews: pd.DataFrame) -> plt.Figure:
    plt.figure(figsize=(10, 6))
    plt.hist(reviews["compound_sentiment"], bins=15, edgecolor='black', color='skyblue')
    plt.xlim([-1, 1])
    plt.xlabel("Compound Sentiment")
    plt.ylabel("Number of Reviews")
    plt.title("Distribution of Compound Sentiment in Reviews")
    plt.show()
    return plt.gcf()

def fig_percentage(reviews: pd.DataFrame) -> plt.Figure:

    sentiment_counts = reviews["sentiment_label"].value_counts()
    total_reviews = sentiment_counts.sum()
    sentiment_percentages = (sentiment_counts / total_reviews * 100).round(2)

    sentiment_analysis = pd.DataFrame({
        "count": sentiment_counts,
        "percentage": sentiment_percentages
    })

    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind="bar", color=["green", "red", "blue"], edgecolor="black")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)

    for index, value in enumerate(sentiment_counts):
        percentage = sentiment_percentages.iloc[index]
        plt.text(index, value + total_reviews * 0.01, f"{percentage}%", ha="center", fontsize=10)

    return plt.gcf()