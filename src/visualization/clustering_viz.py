from sklearn.preprocessing import StandardScaler, QuantileTransformer
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

def fig_feat_distribution(users_feat, platform):
    if platform=='BeerAdvocate':
        color='#E3350F' # red
    elif platform=='RateBeer':
        color='#3E6CE5' # blue

    quantile_transformer = QuantileTransformer(output_distribution='normal')

    nb_features = len(users_feat.columns)

    fig, ax = plt.subplots(nb_features, 6, figsize=(20,4*nb_features))

    for i, col in enumerate(users_feat.columns):
     
        # QQ-Plot for exponential distribution
        stats.probplot(users_feat[col], dist="expon", plot=ax[i][0])
        ax[i][0].set_title("QQ-Plot Exponential")
        ax[i][0].get_lines()[0].set_color(color)
        
        # QQ-Plot for normal distribution
        stats.probplot(users_feat[col], dist="norm", plot=ax[i][1])
        ax[i][1].set_title("QQ-Plot Normal")
        ax[i][1].get_lines()[0].set_color(color)

        # Histogram feature distribution
        sns.histplot(users_feat[col],bins=100, ax=ax[i][2], color=color, legend=False)
        ax[i][2].set_title("Original Distribution")

        # Histogram feature distribution (normalized)
        sns.histplot(StandardScaler().fit_transform(users_feat[[col]]), bins=100, ax=ax[i][3], color=color, legend=False)
        ax[i][3].set_title("Normalized Distribution")

        # Histogram log feature distribution
        sns.histplot(StandardScaler().fit_transform(np.log(1e-5+users_feat[[col]])), bins=100, ax=ax[i][4], color=color, legend=False)
        ax[i][4].set_title("Log Transformed")

        # Histogram quantile transformed feature distribution
        sns.histplot(StandardScaler().fit_transform(
            quantile_transformer.fit_transform(users_feat[[col]])
        ), bins=100, ax=ax[i][5], color=color, legend=False)
        ax[i][5].set_title("Quantile Transformed")

        for hist in ax[i][2:]:
            for patch in hist.patches:  # All histogram patches
                patch.set_facecolor(color)  # Update color

    fig.suptitle(f"{platform} Features Transformation Analysis", fontsize=16, y=1.02)

    plt.tight_layout()
    fig.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
    plt.close()
    return fig

def fig_first_pca(ba_data_pca, rb_data_pca, transform, ba_color='#3E6CE5', rb_color='#E3350F'):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.scatter(
        ba_data_pca[:, 0], 
        ba_data_pca[:, 1], 
        color=ba_color, 
        label='BeerAdvocate', 
        alpha=0.7, 
        edgecolor='k', 
        s=50
    )
    
    ax.scatter(
        rb_data_pca[:, 0], 
        rb_data_pca[:, 1], 
        color=rb_color, 
        label='RateBeer', 
        alpha=0.7, 
        edgecolor='k', 
        s=50
    )
    
    ax.set_title(f"PCA projection with {transform}", fontsize=14)
    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    ax.legend(title="Platform", fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.close()
    return fig

def fig_final_feat_distribution(users_transformed_feat, platform):
    # Set color based on the platform
    if platform == 'BeerAdvocate':
        color = '#E3350F'  # red
    elif platform == 'RateBeer':
        color = '#3E6CE5'  # blue

    # Extract column names and number of features
    columns = users_transformed_feat.columns
    num_columns = len(columns)

    # Create a grid of subplots
    fig, axes = plt.subplots(1, num_columns, figsize=(5 * num_columns, 5), constrained_layout=True)

    # Handle single subplot case
    if num_columns == 1:
        axes = [axes]  # Convert to a list for consistency

    # Add histograms for each feature
    for idx, column in enumerate(columns):
        ax = axes[idx]

        # Plot histogram
        ax.hist(users_transformed_feat[column], bins=20, color=color, alpha=0.6, edgecolor='black')

        # Set log scale for y-axis
        ax.set_yscale('log')

        # Set title and axis labels
        ax.set_title(column)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Add a main title for the figure
    fig.suptitle(f"Transformed and Normalized Features Histograms", fontsize=16)
    plt.close()
    return fig

def fig_clustering_metrics(users_feat, platform, start=2, end=11):
    # Set color based on the platform
    if platform == 'BeerAdvocate':
        color = '#E3350F'  # red
    elif platform == 'RateBeer':
        color = '#3E6CE5'  # blue

    # Elbow Method
    sse = []
    for k in range(start, end):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(users_feat)
        sse.append({"k": k, "sse": kmeans.inertia_})

    sse_df = pd.DataFrame(sse)

    # Silhouette Score
    silhouettes = []
    for k in range(start, end):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(users_feat)
        score = silhouette_score(users_feat, labels)
        silhouettes.append({"k": k, "score": score})

    silhouettes_df = pd.DataFrame(silhouettes)

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Elbow Method
    axes[0].plot(sse_df['k'], sse_df['sse'], marker='o', color=color, label="SSE")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Sum of Squared Errors")
    axes[0].grid(True)

    # Plot Silhouette Score
    axes[1].plot(silhouettes_df['k'], silhouettes_df['score'], marker='o', color=color, label="Silhouette Score")
    axes[1].set_title("Silhouette Score Method")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True)

    # Add a main title
    fig.suptitle("Clustering Metrics Analysis", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close()
    return fig

def fig_second_pca(users_feat, platform):
    # Define color maps based on platform
    if platform == 'BeerAdvocate':
        color_map = {
            'transient': '#FFCCCC',    # Light Red
            'occasional': '#FF6666',   # Medium Red
            'experienced': '#990000'   # Dark Red
        }
    elif platform == 'RateBeer':
        color_map = {
            'transient': '#ADD8E6',    # Light Blue
            'occasional': '#4682B4',   # Medium Blue
            'experienced': '#00008B'  # Dark Blue
        }

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Group data by user type and plot each group
    for user_type, color in color_map.items():
        subset = users_feat[users_feat['users_type'] == user_type]
        ax.scatter(
            subset['PCA_1'], subset['PCA_2'], 
            label=user_type.capitalize(),  # Add labels for the legend
            c=color, alpha=0.7, s=50  # Marker size and transparency
        )

    # Set titles and labels
    ax.set_title(f"{platform} PCA with Clusters", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)

    # Add legend
    ax.legend(title="User Type", loc="best", fontsize=10)

    # Add grid for better readability
    ax.grid(alpha=0.3)
    plt.close()
    return fig

def plot_combined_scatter(ba_clustering_df, rb_clustering_df, features):
    """
    Plots a 2x1 grid (BeerAdvocate and RateBeer) with 3x2 sub-subplots showing scatterplots 
    with clusters in different colors.

    Parameters:
        ba_clustering_df (DataFrame): BeerAdvocate clustering dataset.
        rb_clustering_df (DataFrame): RateBeer clustering dataset.
        features (list of features pair)
    """
    # Color map
    ba_color_map = {
        'occasional': '#FF6666', # Medium Red
        'transient': '#FFCCCC',    # Light Red
        'experienced': '#990000'       # Dark Red
    }

    rb_color_map = {
        'occasional': '#4682B4', # Medium Blue
        'transient': '#ADD8E6',    # Light Blue
        'experienced': '#00008B'       # Dark Blue
    }
    
    # Define dimensions
    n = len(features)
    if n == 1:
        m = 1
    elif (n%2==0):
        m = 2
    else:
        m = 1

    # Create figure and axes
    fig, axes = plt.subplots(int(n/m), int(2*m), figsize=(20, 15))

    # Left column for BeerAdvocate
    for i, (x, y) in enumerate(features):
        row, col = divmod(i, 2)  # Calculate row and column for subplots
        ax = axes[row, col]

        for level, color in ba_color_map.items():
            filtered_data = ba_clustering_df[ba_clustering_df['users_type'] == level]
            sns.scatterplot(data=filtered_data, x=x, y=y, color=color, alpha=0.7, s=20, ax=ax, label=level)

        ax.set_title(f"BA: {x} vs {y}", fontsize=10)
        ax.set_xlabel(x.replace("_", " ").capitalize())
        ax.set_ylabel(y.replace("_", " ").capitalize())
        if x == "total_reviews":
            ax.set_xscale("log")
        if row == 0 and col == 0:  # Add legend only once
            ax.legend()

    # Right column for RateBeer
    for i, (x, y) in enumerate(features):
        row, col = divmod(i, 2)  # Calculate row and column for subplots
        ax = axes[row, col + 2]  # Shift to the right for RateBeer plots
        
        for level, color in rb_color_map.items():
            filtered_data = rb_clustering_df[rb_clustering_df['users_type'] == level]
            sns.scatterplot(data=filtered_data, x=x, y=y, color=color, alpha=0.7, s=20, ax=ax, label=level)

        ax.set_title(f"RB: {x} vs {y}", fontsize=10)
        ax.set_xlabel(x.replace("_", " ").capitalize())
        ax.set_ylabel(y.replace("_", " ").capitalize())
        if x == "total_reviews":
            ax.set_xscale("log")
        if row == 0 and col == 0:  # Add legend only once
            ax.legend()

    plt.close()
    return fig

def plot_scatter_active_time_total_reviews(
    ba_threshold_df, rb_threshold_df, ba_clustering_df, rb_clustering_df
):
    """
    Plots a 2x1 grid (BeerAdvocate and RateBeer) with 3x2 sub-subplots showing combined scatterplots 
    for thresholding (dark gray) and clustering (light gray).

    Parameters:
        ba_threshold_df (DataFrame): BeerAdvocate threshold dataset.
        rb_threshold_df (DataFrame): RateBeer threshold dataset.
        ba_clustering_df (DataFrame): BeerAdvocate clustering dataset.
        rb_clustering_df (DataFrame): RateBeer clustering dataset.
    """
    # Color map
    ba_color_map = {
        'occasional': '#FF6666', # Medium Red
        'transient': '#FFCCCC',    # Light Red
        'experienced': '#990000'       # Dark Red
    }

    rb_color_map = {
        'occasional': '#4682B4', # Medium Blue
        'transient': '#ADD8E6',    # Light Blue
        'experienced': '#00008B'       # Dark Blue
    }

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 1x2 subplots

    # Left column for BeerAdvocate
    
    ax = axes[0]
    
    # Plot clustering
    for level, color in ba_color_map.items():
        filtered_data = ba_clustering_df[ba_clustering_df['users_type'] == level]
        sns.scatterplot(data=filtered_data, x='active_period', y='total_reviews', color=color, alpha=0.7, s=20, ax=ax, label=f"Clustering : {level}")

    # Plot threshold in black
    sns.scatterplot(data=ba_threshold_df, x='active_period', y='total_reviews', color="black", alpha=0.7, s=20, ax=ax, label="Threshold : expert")

    ax.set_title(f"BA groups: Cluster vs Threshold", fontsize=10)
    ax.set_xlabel('Active Period')
    ax.set_ylabel('Total Reviews')
    
    ax.set_yscale("log")
    ax.legend()

    # Right column for RateBeer
    ax = axes[1]
    
    # Plot clustering
    for level, color in rb_color_map.items():
        filtered_data = rb_clustering_df[rb_clustering_df['users_type'] == level]
        sns.scatterplot(data=filtered_data, x='active_period', y='total_reviews', color=color, alpha=0.7, s=20, ax=ax, label=f"Clustering : {level}")

    # Plot threshold in black
    sns.scatterplot(data=rb_threshold_df, x='active_period', y='total_reviews', color="black", alpha=0.7, s=20, ax=ax, label="Threshold : expert")

    ax.set_title(f"RB groups: Cluster vs Threshold", fontsize=10)
    ax.set_xlabel('Active Period')
    ax.set_ylabel('Total Reviews')
    
    ax.set_yscale("log")
    ax.legend()

    plt.close()
    return fig