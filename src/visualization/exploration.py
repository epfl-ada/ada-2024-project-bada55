import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_continent_distribution(df, continent_column='continent') -> Figure:

    # Computation of percentages of continents
    continent_counts = df[continent_column].value_counts().sort_index()
    total_count = continent_counts.sum()
    continent_percentages = (continent_counts / total_count) * 100

    # DataFrame for the plot
    plot_df = pd.DataFrame({
        'Continent': continent_counts.index,
        'Count': continent_counts.values,
        'Percentage': continent_percentages.values
    })

    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count
    sns.barplot(
        x='Continent', y='Count', data=plot_df, ax=ax, color='skyblue', label="Count"
    )

    # Percentages
    for index, row in plot_df.iterrows():
        ax.text(index, row['Count'], f"{row['Percentage']:.1f}%", color='black', ha="center")

    # Titles 
    ax.set_title("Continents")
    ax.set_ylabel("Count")
    ax.set_xlabel("Continent")

    return fig