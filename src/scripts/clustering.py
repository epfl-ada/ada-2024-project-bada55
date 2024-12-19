import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import kstest

import matplotlib.pyplot as plt
from matplotlib.table import Table
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
from plotly.io import write_html

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Data preparation

def data_prep(reviews : pd.DataFrame, users : pd.DataFrame, platform : str):
    # prepare users
    unique_users = reviews['user_id'].unique()
    users_out = users[users['user_id'].isin(unique_users)].sort_values(by='user_id').reset_index(drop=True).copy()
    users_out['platform'] = platform

    # prepare reviews (grouped by user)
    grouped_reviews = reviews.groupby('user_id')
    return (users_out, grouped_reviews)

# User information extraction

def add_total_reviews(users, grouped_reviews):
    users_copy = users.copy()
    users_copy['total_reviews'] = grouped_reviews.agg(total_reviews= ('date', 'count')).reset_index().sort_values(by='user_id').total_reviews
    return users_copy

def add_active_period(users, grouped_reviews):
    users_copy = users.copy()
    first_review = grouped_reviews['date'].min()
    users_copy = users_copy.merge(first_review.rename('first_review_date'), on='user_id', how='left')
    last_review = grouped_reviews['date'].max()
    users_copy = users_copy.merge(last_review.rename('last_review_date'), on='user_id', how='left')
    users_copy['active_period'] = (users_copy['last_review_date'] - users_copy['first_review_date']).dt.days + 1
    users_copy.drop(columns=['first_review_date', 'last_review_date'], inplace=True)
    return users_copy

def add_time_spacing(users, reviews):
    users_copy = users.copy()
    reviews_sorted = reviews.sort_values(['user_id', 'date'])
    time_differences = reviews_sorted.groupby('user_id')['date'].apply(lambda x : x.diff().dt.days)
    users_copy[['mean_time_spacing', 'std_time_spacing']] = time_differences.groupby('user_id').agg(['mean', 'std']).reset_index().sort_values(by= 'user_id')[['mean','std']]
    return users_copy

def add_style_diversity(users, grouped_reviews):
    users_copy = users.copy()
    users_copy['style_diversity'] = grouped_reviews.agg(style_diversity= ('style', 'nunique')).reset_index().sort_values(by= 'user_id').style_diversity
    return users_copy

def add_ratings_std(users, grouped_reviews):
    users_copy = users.copy()
    users_copy['ratings_std'] = grouped_reviews.agg(ratings_std= ('rating', 'std')).reset_index().sort_values(by='user_id').ratings_std
    return users_copy

def extract_users_info(users, reviews, grouped_reviews):
    users_copy = users.copy()
    users_copy = add_total_reviews(users_copy, grouped_reviews)
    users_copy = add_active_period(users_copy, grouped_reviews)
    users_copy = add_time_spacing(users_copy, reviews)
    users_copy = add_style_diversity(users_copy, grouped_reviews)
    users_copy = add_ratings_std(users_copy, grouped_reviews)
    
    return users_copy

# Features Definition

def features_implementation(users_info, selected_feat):
    columns = ['user_id', 'platform']
    for col in selected_feat:
        columns.append(col)
    
    users_feat = users_info[columns].dropna()
    users_feat = users_feat.drop(columns='platform').set_index('user_id')

    return users_feat

# Show features distribution (with and without transformation)

def fig_feat_distribution(users_feat, platform, color):
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

    return (fig, ax)

def feat_transform_normalize(users_feat, transform='log'):
    if transform=='log':
        transformed_feat = np.log(1e-5+users_feat)
    
    elif transform=='quantile':
        quantile_transformer = QuantileTransformer(output_distribution='normal')
        transformed_feat = quantile_transformer.fit_transform(users_feat)
    
    else:
        print("Non valid transform selection. Try 'log' or 'quantile'")
        transformed_feat = []

    #transformed_feat = pd.DataFrame(transformed_feat, index=users_feat.index, columns=users_feat)
    
    return transformed_feat


def pca(users_feat, transform='log'):
    transformed_feat = feat_transform_normalize(users_feat, transform)
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(StandardScaler().fit_transform(transformed_feat))
    
    return data_pca

def fig_first_pca(ba_users_feat, rb_users_feat, transform='log', ba_color='#3E6CE5', rb_color='#E3350F'):
    ba_data_pca = pca(ba_users_feat, transform)
    rb_data_pca = pca(rb_users_feat, transform)

    fig_pca = go.Figure()

    # Add BeerAdvocate data
    fig_pca.add_trace(go.Scatter(
        x=ba_data_pca[:, 0], 
        y=ba_data_pca[:, 1],
        mode='markers',
        name='BeerAdvocate',
        marker=dict(color=ba_color)
    ))

    # Add RateBeer data
    fig_pca.add_trace(go.Scatter(
        x=rb_data_pca[:, 0], 
        y=rb_data_pca[:, 1],
        mode='markers',
        name='RateBeer',
        marker=dict(color=rb_color)
    ))

    # Set the title and layout
    fig_pca.update_layout(
        title=f"PCA projection with {transform}",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend=dict(title="Platform")
    )

    return fig_pca

# Clustering

def fig_final_feat_distribution(users_transformed_feat, color):
    #users_transformed_feat = pd.DataFrame(users_transformed_feat, index=users_feat.index, columns=users_feat)

    columns = users_transformed_feat.columns

    fig = sp.make_subplots(rows=1, cols=len(columns), subplot_titles=columns)

    # Add histograms to each subplot
    for idx, column in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=users_transformed_feat[column], nbinsx=20, opacity=0.6, marker=dict(color=color)),
            row=1, col=idx + 1
        )
        # Update log scale for y-axis
        fig.update_yaxes(type="log", row=1, col=idx + 1)

    # Update layout
    fig.update_layout(
        height=300, width=1000, showlegend=False, title_text=f"Transformed and normalized features histograms"
    )

    return fig

def fig_clustering_metrics(users_feat, color, start=2, end=11):
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
        print(f"Implementation with k={k} done")

    silhouettes_df = pd.DataFrame(silhouettes)

    # Create subplots
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=["Elbow Method", "Silhouette Score Method"])

    # Plot Elbow Method
    fig.add_trace(
        go.Scatter(x=sse_df['k'], y=sse_df['sse'], mode='lines+markers', name="SSE", marker=dict(color=color)),
        row=1, col=1
    )
    fig.update_xaxes(title_text="K", row=1, col=1)
    fig.update_yaxes(title_text="Sum of Squared Errors", row=1, col=1)

    # Plot Silhouette Score
    fig.add_trace(
        go.Scatter(x=silhouettes_df['k'], y=silhouettes_df['score'], mode='lines+markers', name="Silhouette Score", marker=dict(color=color)),
        row=1, col=2
    )
    fig.update_xaxes(title_text="K", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    # Update layout
    fig.update_layout(
        title_text="Clustering Metrics Analysis",
        height=500, width=1000,
        showlegend=False
    )

    return fig

def clustering(features, nb_clusters):
    columns = features.columns
    scaled_features = pd.DataFrame(StandardScaler().fit(features).transform(features), columns= columns)

    kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    feat_cluster = pd.DataFrame(pd.Series(labels, index= features.index, name='cluster')).reset_index()
    feat_cluster['user_id'] = feat_cluster['user_id'].astype(str)
    return feat_cluster

def label_definition(users, clust_feat, selected_feat):
    clust_users = clust_feat.join(users.set_index('user_id'), on='user_id', how='left')

    cluster_mean_total_reviews = clust_users.groupby('cluster')['total_reviews'].mean()
    sorted_tr_clusters = cluster_mean_total_reviews.sort_values().index
    cluster_mean_time_spacing = clust_users.groupby('cluster')['mean_time_spacing'].mean()
    sorted_ts_clusters = cluster_mean_time_spacing.sort_values().index

    cluster_labels = {sorted_ts_clusters[0]: 'transient', 
                    sorted_ts_clusters[2]: 'occasional', 
                    sorted_tr_clusters[2]: 'experienced'}

    clust_users['users_type'] = clust_users['cluster'].map(cluster_labels)

    columns = ['users_type']
    for col in selected_feat:
        columns.append(col)
    users_clust_feat = users.copy()
    users_clust_feat = clust_users[columns]

    cluster_summary = users_clust_feat.groupby('users_type').agg(['mean','median','std','min','max'])

    return (clust_users, users_clust_feat, cluster_summary)
    

# Visualization

def add_pca(clust_users, selected_feat):
    data_for_pca = clust_users[selected_feat]
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(np.log(1e-5+data_for_pca)))
    clust_users_pca = clust_users.copy()
    clust_users_pca['PCA_1'] = pca_result[:,0]
    clust_users_pca['PCA_2'] = pca_result[:,1]
    return clust_users_pca

def fig_second_pca(users_feat, platform):
    if platform=='BeerAdvocate':
        color_map = {
            'transient': '#ADD8E6',    # Light Blue
            'occasional': '#4682B4', # Medium Blue
            'experienced': '#00008B'       # Dark Blue
        }
    elif platform=='RateBeer':
        color_map = {
            'transient': '#FFCCCC',    # Light Red
            'occasional': '#FF6666', # Medium Red
            'experienced': '#990000'       # Dark Red
        }
    
    fig_pca = px.scatter(
        users_feat,
        x='PCA_1',
        y='PCA_2',
        color='users_type',
        color_discrete_map=color_map,
        title="BA PCA with Clusters",
        labels={'PCA_1': 'Principal Component 1', 'PCA_2': 'Principal Component 2'},
        hover_data=['user_id', 'platform', 'total_reviews']  # Optional, displays additional data on hover
    )

    return fig_pca



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
        'occasional': '#4682B4', # Medium Blue
        'transient': '#ADD8E6',    # Light Blue
        'experienced': '#00008B'       # Dark Blue
    }

    rb_color_map = {
        'occasional': '#FF6666', # Medium Red
        'transient': '#FFCCCC',    # Light Red
        'experienced': '#990000'       # Dark Red
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

    return (fig, axes)



# Cluster VS Threshold


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
        'occasional': '#4682B4', # Medium Blue
        'transient': '#ADD8E6',    # Light Blue
        'experienced': '#00008B'       # Dark Blue
    }

    rb_color_map = {
        'occasional': '#FF6666', # Medium Red
        'transient': '#FFCCCC',    # Light Red
        'experienced': '#990000'       # Dark Red
    }

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))  # 1x2 subplots

    # Left column for BeerAdvocate
    
    ax = axes[0]
    
    # Plot clustering
    for level, color in ba_color_map.items():
        filtered_data = ba_clustering_df[ba_clustering_df['users_type'] == level]
        sns.scatterplot(data=filtered_data, x='active_period', y='total_reviews', color=color, alpha=0.7, s=20, ax=ax, label="Clustering : {level}")

    # Plot threshold in black
    sns.scatterplot(data=ba_threshold_df, x='active_period', y='total_reviews', color="black", alpha=0.7, s=20, ax=ax, label="Threshold")

    ax.set_title(f"BA: Active Period vs Total Reviews", fontsize=10)
    ax.set_xlabel('Active Period')
    ax.set_ylabel('Total Reviews')
    
    ax.set_yscale("log")
    ax.legend()

    # Right column for RateBeer
    ax = axes[1]
    
    # Plot clustering
    for level, color in rb_color_map.items():
        filtered_data = rb_clustering_df[rb_clustering_df['users_type'] == level]
        sns.scatterplot(data=filtered_data, x='active_period', y='total_reviews', color=color, alpha=0.7, s=20, ax=ax, label="Clustering : {level}")

    # Plot threshold in black
    sns.scatterplot(data=rb_threshold_df, x='active_period', y='total_reviews', color="black", alpha=0.7, s=20, ax=ax, label="Threshold")

    ax.set_title(f"RB: Active Period vs Total Reviews", fontsize=10)
    ax.set_xlabel('Active Period')
    ax.set_ylabel('Total Reviews')
    
    ax.set_yscale("log")
    ax.legend()

    # Adjust layout
    return (fig, axes)



def table_clusterVSthreshold(ba_experts_threshold, rb_experts_threshold, ba_experts_clustering, rb_experts_clustering):

    # Données
    headers = ["", "BeerAdvocate", "  RateBeer  "]
    rows = [
        ["Number of experienced users with Clustering method", len(ba_experts_clustering), len(rb_experts_clustering)],
        ["Number of experienced users with Threshold method", len(ba_experts_threshold), len(rb_experts_threshold)],
        [
            "Percentage of experienced users from Threshold in group from Clustering", 
            f"{(len(set(ba_experts_threshold['user_id']).intersection(set(ba_experts_clustering['user_id']))) / len(ba_experts_threshold) * 100):.2f}%",
            f"{(len(set(rb_experts_threshold['user_id']).intersection(set(rb_experts_clustering['user_id']))) / len(rb_experts_threshold) * 100):.2f}%"
        ]
    ]

    # Création du tableau
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('tight')
    ax.axis('off')

    #plt.title("Proportion of experienced users from threshold method ", fontsize=25)

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center', bbox=[0, 0, 1.5, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(headers))))

    return (fig, ax)































def plot_clustering_metrics(features_X, start=2, end=11):
    # Elbow Method
    sse = []
    for k in range(start, end):
        kmeans = KMeans(n_clusters=k, random_state=10).fit(features_X)
        sse.append({"k": k, "sse": kmeans.inertia_})

    sse = pd.DataFrame(sse)

    # Silhouette Score
    silhouettes = []
    for k in range(start, end):
        labels = KMeans(n_clusters=k, random_state=10).fit_predict(features_X)
        score = silhouette_score(features_X, labels)
        silhouettes.append({"k": k, "score": score})
        print(f"Implementation with k={k} done")

    silhouettes = pd.DataFrame(silhouettes)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Elbow Method
    axes[0].plot(sse.k, sse.sse, marker='o')
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Sum of Squared Errors")
    axes[0].set_title("Elbow Method")

    # Plot Silhouette Score
    axes[1].plot(silhouettes.k, silhouettes.score, marker='o')
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score Method")

    # Adjust layout
    plt.tight_layout()
    plt.show()

def threshold_df(features, dict_thresholds_up, dict_thresholds_down):
    threshold_feat = features.copy()
    for key, value in dict_thresholds_up.items():
        threshold_feat[threshold_feat[key]] >= value

    for key, value in dict_thresholds_down.items():
        threshold_feat[threshold_feat[key]] < value    
    
    return threshold_feat