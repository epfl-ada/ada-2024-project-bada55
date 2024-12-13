import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

def data_prep(reviews : pd.DataFrame, users : pd.DataFrame, platform : str):
    # prepare users
    unique_users = reviews['user_id'].unique()
    users_out = users[users['user_id'].isin(unique_users)].sort_values(by='user_id').reset_index(drop=True).copy()
    users_out['platform'] = platform

    # prepare reviews (grouped by user)
    grouped_reviews = reviews.groupby('user_id')
    return (users_out, grouped_reviews)

def add_feat_total_reviews(grouped_reviews):
    return grouped_reviews.agg(total_reviews= ('date', 'count')).reset_index().sort_values(by='user_id').total_reviews

def add_feat_time_spacing():
    return

def add_feat_style_diversity(grouped_reviews):
    return grouped_reviews.agg(style_diversity= ('style', 'nunique')).reset_index().sort_values(by='user_id').style_diversity

def add_feat_ratings_std(grouped_reviews):
    return grouped_reviews.agg(ratings_std= ('rating', 'std')).reset_index().sort_values(by='user_id').ratings_std


def features_implementation(users : pd.DataFrame, reviews : pd.DataFrame, platform : str):
    (users_feat, grouped_reviews) = data_prep(users, reviews, platform)

    # Define features
    users_feat['total_reviews'] = add_feat_total_reviews(grouped_reviews)
    users_feat[['mean_time_spacing', 'std_time_spacing']] = add_feat_time_spacing()
    users_feat['style_diversity'] = add_feat_style_diversity(grouped_reviews)

    # Clean features df
    list_to_drop = ['joined', 'first_review_date', 'last_review_date', 'ratings_std']
    users_feat = users_feat.dropna().drop(columns=list_to_drop).set_index('user_id').copy() 

    return users_feat

def clustering(features, nb_clusters):
    columns = features.columns
    scaled_features = pd.DataFrame(StandardScaler().fit(features).transform(features), columns= columns)

    kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    feat_cluster = pd.DataFrame(pd.Series(labels, index= features.index, name='cluster')).reset_index()
    return feat_cluster

def generate_list_best_users(feat_cluster, platform):
    # Mean of each features -> choose higher mean for total reviews
    cluster_mean = feat_cluster.drop(columns='user_id').groupby('cluster').mean()
    exp_cluster = cluster_mean['total_reviews'].idxmax()
    
    list_exp = feat_cluster[feat_cluster['cluster'] == exp_cluster]['user_id']
    list_exp.to_csv(f"../generated/{platform}_experts.csv", index=False)
    


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