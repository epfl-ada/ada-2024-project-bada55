import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import kstest

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp

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
    
    users_feat = users_info[columns]
    users_feat.dropna(inplace=True)
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
        y=-ba_data_pca[:, 1],
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

    cluster_mean = clust_users.groupby('cluster')['total_reviews'].mean()
    sorted_clusters = cluster_mean.sort_values().index
    cluster_labels = {sorted_clusters[0]: 'beginners', 
                    sorted_clusters[1]: 'intermediate', 
                    sorted_clusters[2]: 'experienced'}

    clust_users['experience_level'] = clust_users['cluster'].map(cluster_labels)

    columns = ['experience_level']
    for col in selected_feat:
        columns.append(col)
    users_clust_feat = users.copy()
    users_clust_feat = clust_users[columns]

    cluster_summary = users_clust_feat.groupby('experience_level').agg(['mean','median','std','min','max'])

    return (clust_users, users_clust_feat, cluster_summary)
    

# Visualization

def add_pca(clust_users, selected_feat):
    data_for_pca = clust_users[selected_feat]
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_for_pca)
    clust_users_pca = clust_users.copy()
    clust_users_pca['PCA_1'] = pca_result[:,0]
    clust_users_pca['PCA_2'] = pca_result[:,1]
    return clust_users_pca

def fig_second_pca(users_feat, platform):
    if platform=='BeerAdvocate':
        color_map = {
            'beginners': '#ADD8E6',    # Light Blue
            'intermediate': '#4682B4', # Medium Blue
            'experienced': '#00008B'       # Dark Blue
        }
    elif platform=='RateBeer':
        color_map = {
            'beginners': '#FFCCCC',    # Light Red
            'intermediate': '#FF6666', # Medium Red
            'experienced': '#990000'       # Dark Red
        }
    
    fig_pca = px.scatter(
        users_feat,
        x='PCA_1',
        y='PCA_2',
        color='experience_level',
        color_discrete_map=color_map,
        title="BA PCA with Clusters",
        labels={'PCA_1': 'Principal Component 1', 'PCA_2': 'Principal Component 2'},
        hover_data=['user_id', 'platform', 'total_reviews']  # Optional, displays additional data on hover
    )

    return fig_pca


import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.io import write_html

def generate_user_scatterplots(ba_data, rb_data, output_file="clust_feat_scatterplots.html"):
    """
    Generate a single HTML file with scatter plot subplots comparing BA and RB user clusters.
    
    Args:
        ba_data (DataFrame): DataFrame for BeerAdvocate user clusters.
        rb_data (DataFrame): DataFrame for RateBeer user clusters.
        output_file (str): Path to save the HTML file.
    
    Returns:
        fig
    """
    # Color map
    color_map = {
        'beginners': '#F5B7B1', # light pink 
        'intermediate': '#F4D03F', # yellow
        'experienced': '#196F3D' # deep green
    }

    # Initialize subplots
    fig = sp.make_subplots(
        rows=2, cols=4,
        subplot_titles=[
            "BA: Active Period vs Total Reviews",
            "RB: Active Period vs Total Reviews",
            "BA: Total Reviews vs Style Diversity",
            "RB: Total Reviews vs Style Diversity",
            "BA: Active Period vs Mean Time Spacing",
            "RB: Active Period vs Mean Time Spacing",
            "BA: Active Period vs Std Time Spacing",
            "RB: Active Period vs Std Time Spacing"
        ]
    )

    # Define subplot configurations
    plot_configs = [
        (ba_data, 1, 1, 'active_period', 'total_reviews', "log"),
        (rb_data, 2, 1, 'active_period', 'total_reviews', "log"),
        (ba_data, 1, 2, 'total_reviews', 'style_diversity', "log"),
        (rb_data, 2, 2, 'total_reviews', 'style_diversity', "log"),
        (ba_data, 1, 3, 'active_period', 'mean_time_spacing', None),
        (rb_data, 2, 3, 'active_period', 'mean_time_spacing', None),
        (ba_data, 1, 4, 'active_period', 'std_time_spacing', None),
        (rb_data, 2, 4, 'active_period', 'std_time_spacing', None)
    ]

    # Add scatter plots to subplots
    for data, row, col, x, y, x_axis_type in plot_configs:
        for level, color in color_map.items():
            filtered_data = data[data['experience_level'] == level]
            fig.add_trace(
                go.Scatter(
                    x=filtered_data[x],
                    y=filtered_data[y],
                    mode='markers',
                    marker=dict(color=color, size=6),
                    name=level.capitalize(),
                    legendgroup=level,
                    showlegend=(row == 1 and col == 1)  # Show legend only once
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text=x.replace('_', ' ').title(), row=row, col=col, type=x_axis_type)
        fig.update_yaxes(title_text=y.replace('_', ' ').title(), row=row, col=col)

    # Update layout
    fig.update_layout(
        title_text="BA vs RB User Cluster Comparison",
        height=1000,
        width=2500,
        showlegend=True
    )

    # Save the figure as an HTML file
    write_html(fig, file=output_file, auto_open=False)

    return fig



































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