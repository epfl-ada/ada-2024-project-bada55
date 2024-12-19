import matplotlib.pyplot as plt
import seaborn as sns

def plot_dominant_topics_heatmap(topic_distribution):
    """
    Plot a heatmap of dominant topics by dataset.

    Parameters:
    topic_distribution (DataFrame): A pandas DataFrame where rows represent datasets 
                                    (e.g., 'BeerAdvocate', 'RateBeer') and columns represent topics.

    Returns:
    matplotlib.figure.Figure: The generated heatmap figure.
    """
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize topic distribution and create the heatmap
    normalized_distribution = (topic_distribution / topic_distribution.loc['BeerAdvocate'].sum() * 100).round(2)
    sns.heatmap(
        normalized_distribution,
        annot=True,
        cmap="YlGnBu",
        cbar_kws={'label': 'Percentage'},
        fmt='.2f',
        ax=ax
    )

    # Set plot labels and title
    ax.set_title("Heatmap of Dominant Topics by Dataset")
    ax.set_xlabel("Dominant Topic")
    ax.set_ylabel("Dataset")

    # Adjust layout
    plt.tight_layout()
    plt.close()
    return fig
