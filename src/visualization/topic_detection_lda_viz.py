import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def fig_heatmap_topic_distribution_dataset(topic_distribution: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = ax.imshow(topic_distribution, cmap="YlGnBu", aspect="auto")
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Topic Count", rotation=270, labelpad=15)
    for i in range(topic_distribution.shape[0]):
        for j in range(topic_distribution.shape[1]):
            value = topic_distribution.iloc[i, j]
            color = "white" if value in [4524, 3793] else "black"  # Highlight specific values
            font_size = max(10, min(25, value / 50))  # Scale font size between 10 and 25
            ax.text(j, i, str(value), ha="center", va="center", 
                        color=color, fontsize=font_size)
    ax.set_xticks(np.arange(topic_distribution.shape[1]))
    ax.set_yticks(np.arange(topic_distribution.shape[0]))
    ax.set_xticklabels(topic_distribution.columns)
    ax.set_yticklabels(topic_distribution.index)
    ax.set_xlabel("Dominant Topic")
    ax.set_ylabel("Dataset")
    ax.set_title("Topic Distribution by Dataset")
    
    plt.tight_layout()
    plt.close()
    return fig