import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fig_comp_coeff_topic(coeff_ba: pd.Series, coeff_rb: pd.Series):
    topics = ['appearance', 'aroma', 'palate', 'taste', 'overall']
    coeff_ba_norm = coeff_ba / coeff_ba.sum() * 100
    coeff_rb_norm = coeff_rb / coeff_rb.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.5
    x = np.arange(2)

    bottom_ba = 0
    bottom_rb = 0
    for i, topic in enumerate(topics):
        ax.bar(
            x[0], coeff_ba_norm[i], width, 
            label=topic, 
            bottom=bottom_ba, color=f'C{i}'
        )
        ax.text(
            x[0], bottom_ba + coeff_ba_norm[i] / 2, 
            f"{coeff_ba_norm[i]:.1f}%", ha='center', va='center', fontsize=10
        )
        bottom_ba += coeff_ba_norm[i]

        ax.bar(
            x[1], coeff_rb_norm[i], width, 
            bottom=bottom_rb, color=f'C{i}'
        )
        ax.text(
            x[1], bottom_rb + coeff_rb_norm[i] / 2, 
            f"{coeff_rb_norm[i]:.1f}%", ha='center', va='center', fontsize=10
        )
        bottom_rb += coeff_rb_norm[i]

    ax.set_title("Stacked Bar Chart of Coefficients by Topic (Normalized to 100%)", fontsize=14)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['BeerAdvocate', 'RateBeer'], fontsize=12)
    ax.legend(title="Topics", loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.close()
    return fig




def fig_pred_vs_reel(
        ba_reviews: pd.DataFrame,
        rb_reviews: pd.DataFrame,
        predict_ba: callable,
        predict_rb: callable,
        r2_ba: float,
        r2_rb: float,
):
    def normalize(df):
        return (df - df.min()) / (df.max() - df.min())
    
    y_ba = normalize(ba_reviews[["appearance", "aroma", "palate", "taste", "overall", "rating"]])
    y_rb = normalize(rb_reviews[["appearance", "aroma", "palate", "taste", "overall", "rating"]])

    y_real_ba = y_ba["rating"]
    y_pred_ba = predict_ba(y_ba[['appearance', 'aroma', 'palate', 'taste', 'overall']])
    y_real_rb = y_rb["rating"]
    y_pred_rb = predict_rb(y_rb[['appearance', 'aroma', 'palate', 'taste', 'overall']])

    range_ba = np.random.choice(len(y_real_ba), int(0.001 * len(y_real_ba)), replace=False)
    y_real_ba = y_real_ba.iloc[range_ba]
    y_pred_ba = y_pred_ba.iloc[range_ba]

    range_rb = np.random.choice(len(y_real_rb), int(0.001 * len(y_real_rb)), replace=False)
    y_real_rb = y_real_rb.iloc[range_rb]
    y_pred_rb = y_pred_rb.iloc[range_rb]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y_real_ba, y_pred_ba, c='blue', alpha=0.7, label='BeerAdvocate Predictions')
    axes[0].plot([y_real_ba.min(), y_real_ba.max()], [y_real_ba.min(), y_real_ba.max()], 'r--', label='y = x')
    axes[0].set_title(f"BeerAdvocate R² = {r2_ba:.2f}", fontsize=12)
    axes[0].set_xlabel("Real Rating", fontsize=10)
    axes[0].set_ylabel("Predicted Rating", fontsize=10)
    axes[0].legend()

    axes[1].scatter(y_real_rb, y_pred_rb, c='green', alpha=0.7, label='RateBeer Predictions')
    axes[1].plot([y_real_rb.min(), y_real_rb.max()], [y_real_rb.min(), y_real_rb.max()], 'r--', label='y = x')
    axes[1].set_title(f"RateBeer R² = {r2_rb:.2f}", fontsize=12)
    axes[1].set_xlabel("Real Rating", fontsize=10)
    axes[1].set_ylabel("Predicted Rating", fontsize=10)
    axes[1].legend()

    fig.suptitle("Predicted vs. Real Ratings", fontsize=14)
    plt.tight_layout()
    plt.close()
    return fig