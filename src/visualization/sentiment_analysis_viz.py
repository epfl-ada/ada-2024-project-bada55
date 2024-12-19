import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def fig_distribution_sentiment(reviews: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        reviews["positive_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Positive Sentiment", color='green'
    )
    ax.hist(
        reviews["negative_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Negative Sentiment", color='salmon'
    )
    ax.hist(
        reviews["neutral_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Neutral Sentiment", color='blue'
    )
    ax.set_xlim([0, 1])
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Number of Reviews")
    ax.set_title(f"Distribution of Positive, Negative, and Neutral Sentiment in Reviews of {title}")
    ax.legend()
    plt.close()
    return fig

def fig_compound_sentiment(reviews: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        reviews["compound_sentiment"], 
        bins=15, 
        edgecolor='black', 
        color='skyblue'
    )
    ax.set_xlim([-1, 1])
    ax.set_xlabel("Compound Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_title(f"Distribution of Compound Sentiment in Reviews of {title}")
    plt.close()
    return fig

def fig_percentage(reviews: pd.DataFrame, title: str) -> plt.Figure:

    sentiment_counts = reviews["sentiment_label"].value_counts()
    total_reviews = sentiment_counts.sum()
    sentiment_percentages = (sentiment_counts / total_reviews * 100).round(2)

    sentiment_analysis = pd.DataFrame({
        "count": sentiment_counts,
        "percentage": sentiment_percentages
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_counts.plot(
        kind="bar", 
        color=["green", "red", "blue"], 
        edgecolor="black", 
        ax=ax
    )
    ax.set_title(f"Sentiment Distribution in {title}")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticklabels(sentiment_counts.index, rotation=0)

    for index, value in enumerate(sentiment_counts):
        percentage = sentiment_percentages.iloc[index]
        ax.text(index, value + total_reviews * 0.01, f"{percentage}%", ha="center", fontsize=10)

    plt.close()
    return fig

def fig_sentiment_distribution(beeradvocate_sizes, ratebeer_sizes, labels):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    colors = ["green", "red", "blue"]

    axes[0].pie(
        beeradvocate_sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    axes[0].set_title("BeerAdvocate Sentiment Distribution")

    axes[1].pie(
        ratebeer_sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    axes[1].set_title("RateBeer Sentiment Distribution")

    plt.suptitle("Comparison of Sentiment Distribution Between BeerAdvocate and RateBeer")

    plt.tight_layout()

    plt.close()

    return fig

def fig_distribution_sentiment_combined(
    ba_reviews: pd.DataFrame, 
    rb_reviews: pd.DataFrame, 
    ba_title: str, 
    rb_title: str
) -> plt.Figure:
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].hist(
        ba_reviews["positive_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Positive", color='green'
    )
    axes[0].hist(
        ba_reviews["negative_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Negative", color='salmon'
    )
    axes[0].hist(
        ba_reviews["neutral_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Neutral", color='blue'
    )
    axes[0].set_xlim([0, 1])
    axes[0].set_xlabel("Sentiment Score")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_title(f"Distribution of Sentiments in {ba_title}")
    axes[0].legend()

    axes[1].hist(
        rb_reviews["positive_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Positive", color='green'
    )
    axes[1].hist(
        rb_reviews["negative_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Negative", color='salmon'
    )
    axes[1].hist(
        rb_reviews["neutral_sentiment"], bins=15, alpha=0.5, edgecolor='black', label="Neutral", color='blue'
    )
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel("Sentiment Score")
    axes[1].set_title(f"Distribution of Sentiments in {rb_title}")
    axes[1].legend()

    fig.tight_layout()
    plt.close()
    return fig