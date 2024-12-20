import matplotlib.pyplot as plt
import seaborn as sns

def plot_dominant_topics_heatmap(topic_distribution):

    fig, ax = plt.subplots(figsize=(10, 6))

    normalized_distribution = (topic_distribution / topic_distribution.loc['BeerAdvocate'].sum() * 100).round(2)
    sns.heatmap(
        normalized_distribution,
        annot=True,
        cmap="YlGnBu",
        cbar_kws={'label': 'Percentage'},
        fmt='.2f',
        ax=ax
    )

    ax.set_title("Heatmap of Dominant Topics by Dataset")
    ax.set_xlabel("Dominant Topic")
    ax.set_ylabel("Dataset")

    plt.tight_layout()
    plt.close()
    return fig
