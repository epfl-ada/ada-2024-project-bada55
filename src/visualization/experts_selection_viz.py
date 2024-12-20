import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def fig_exp_site(data_exp_ba: np.array, data_exp_rb: np.array):
    categories = ['Expert', 'Intermediate', 'Novice', 'Debutant']
    bars = ['Users (%)', 'Reviews (%)']

    data1_df = pd.DataFrame(data_exp_ba, columns=categories, index=bars)
    data2_df = pd.DataFrame(data_exp_rb, columns=categories, index=bars)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    fig.suptitle("Comparison of Coefficients by Category for Both Datasets", fontsize=14)

    ax = axes[0]
    data1_df.plot(kind='bar', stacked=True, ax=ax, color=['#FF6666', '#FFA07A', '#FFD700', '#98FB98'], edgecolor='black')
    ax.set_title("BeerAdvocate", fontsize=12)
    ax.set_xlabel("Category", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_xticklabels(bars, rotation=0)

    ax = axes[1]
    data2_df.plot(kind='bar', stacked=True, ax=ax, color=['#4682B4', '#5F9EA0', '#7FFFD4', '#B0C4DE'], edgecolor='black')
    ax.set_title("RateBeer", fontsize=12)
    ax.set_xlabel("Category", fontsize=10)
    ax.set_xticklabels(bars, rotation=0)

    for ax, data in zip(axes, [data1_df, data2_df]):
        for bar_group, category in zip(ax.containers, categories):
            for bar in bar_group:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2,
                            f"{100*height:.3f}%", ha='center', va='center', fontsize=8, color='black')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close()
    return fig