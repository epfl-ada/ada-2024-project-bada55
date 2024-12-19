import pandas as pd
from src.visualization.clustering_viz import *
from sklearn.decomposition import PCA


def data_prep(reviews : pd.DataFrame, users : pd.DataFrame, platform : str):
    # prepare users
    users = users[['user_id', 'joined']]
    users['user_id'] = users['user_id'].astype(str)
    reviews['user_id'] = reviews['user_id'].astype(str)

    unique_users = reviews['user_id'].unique()
    users_out = users[users['user_id'].isin(unique_users)].sort_values(by='user_id').reset_index(drop=True).copy()
    users_out['platform'] = platform

    # prepare reviews (grouped by user)
    grouped_reviews = reviews.groupby('user_id')
    return (users_out, grouped_reviews)

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
    reviews['date'] = pd.to_datetime(reviews['date'])
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

def features_implementation(users_info, selected_feat):
    columns = ['user_id', 'platform']
    for col in selected_feat:
        columns.append(col)
    
    users_feat = users_info[columns].dropna()
    users_feat = users_feat.drop(columns='platform').set_index('user_id')

    return users_feat

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
    
def add_pca(clust_users, selected_feat):
    data_for_pca = clust_users[selected_feat]
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(np.log(1e-5+data_for_pca)))
    clust_users_pca = clust_users.copy()
    clust_users_pca['PCA_1'] = pca_result[:,0]
    clust_users_pca['PCA_2'] = pca_result[:,1]
    return clust_users_pca

def clustering_fig(
        ba_reviews: pd.DataFrame, 
        rb_reviews: pd.DataFrame,
        ba_reviews_experts: pd.DataFrame, 
        rb_reviews_experts: pd.DataFrame,
        ba_users: pd.DataFrame,
        rb_users: pd.DataFrame,
        ba_platform: str,
        rb_platform: str,
        ):

    figs = []

    (ba_users, ba_grouped_reviews) = data_prep(ba_reviews, ba_users, 'BeerAdvocate')
    (rb_users, rb_grouped_reviews) = data_prep(rb_reviews, rb_users, 'RateBeer')

    ba_users = extract_users_info(ba_users, ba_reviews, ba_grouped_reviews)
    rb_users = extract_users_info(rb_users, rb_reviews, rb_grouped_reviews)

    selected_feat = ['total_reviews', 'mean_time_spacing', 'std_time_spacing', 'style_diversity']

    ba_features = features_implementation(ba_users, selected_feat)
    rb_features = features_implementation(rb_users, selected_feat)
    
    fig_ba = fig_feat_distribution(ba_features, 'BeerAdvocate')
    fig_rb = fig_feat_distribution(rb_features, 'RateBeer')

    figs.append(fig_ba)
    figs.append(fig_rb)

    transform = 'log'

    ba_data_pca = pca(ba_features, transform)
    rb_data_pca = pca(rb_features, transform)
    fig = fig_first_pca(ba_data_pca, rb_data_pca, transform)
    figs.append(fig)

    transform = 'quantile'

    ba_data_pca = pca(ba_features, transform)
    rb_data_pca = pca(rb_features, transform)
    fig = fig_first_pca(ba_data_pca, rb_data_pca, transform)
    figs.append(fig)

    ba_features.drop(index=ba_features[ba_features['std_time_spacing']==0].index, inplace=True)
    rb_features.drop(index=rb_features[rb_features['std_time_spacing']==0].index, inplace=True)

    transform = 'log'
    ba_transformed_feat = feat_transform_normalize(ba_features, transform='log')
    rb_transformed_feat = feat_transform_normalize(rb_features, transform='log')

    fig_ba = fig_final_feat_distribution(ba_transformed_feat, 'BeerAdvocate')
    fig_rb = fig_final_feat_distribution(rb_transformed_feat, 'RateBeer')

    figs.append(fig_ba)
    figs.append(fig_rb)

    fig_ba = fig_clustering_metrics(ba_transformed_feat, 'BeerAdvocate', end=8)
    fig_rb = fig_clustering_metrics(rb_transformed_feat, 'RateBeer', end=8)

    figs.append(fig_ba)
    figs.append(fig_rb)

    nb_clusters = 3

    ba_clust_features = clustering(ba_transformed_feat, nb_clusters)
    rb_clust_features = clustering(rb_transformed_feat, nb_clusters)

    (ba_clust_users, ba_clust_feat, ba_clust_summary) = label_definition(ba_users, ba_clust_features, selected_feat)
    (rb_clust_users, rb_clust_feat, rb_clust_summary) = label_definition(rb_users, rb_clust_features, selected_feat)

    ba_clust_users = add_pca(ba_clust_users, selected_feat)
    rb_clust_users = add_pca(rb_clust_users, selected_feat)

    fig_ba = fig_second_pca(ba_clust_users, 'BeerAdvocate')
    fig_rb = fig_second_pca(rb_clust_users, 'RateBeer')

    figs.append(fig_ba)
    figs.append(fig_rb)

    features = [
        ("total_reviews", "style_diversity"),
        ("total_reviews", "mean_time_spacing"),
        ("total_reviews", "std_time_spacing"),
        ("mean_time_spacing", "style_diversity"),
        ("std_time_spacing", "style_diversity"),
        ("mean_time_spacing", "std_time_spacing"),
    ]

    fig = plot_combined_scatter(ba_clust_users, rb_clust_users, features)
    figs.append(fig)

    ba_experts_id_threshold = ba_reviews_experts.copy()
    rb_experts_id_threshold = rb_reviews_experts.copy()


    columns = ['user_id', 'active_period']
    selected_feat = ['total_reviews', 'style_diversity', 'mean_time_spacing', 'std_time_spacing']
    for col in selected_feat:
        columns.append(col)

    ba_experts_threshold = ba_clust_users[ba_clust_users['user_id'].isin(ba_experts_id_threshold['user_id'])]
    rb_experts_threshold = rb_clust_users[rb_clust_users['user_id'].isin(rb_experts_id_threshold['user_id'])]
    ba_experts_clustering = ba_clust_users[ba_clust_users['users_type'] == 'experienced']
    rb_experts_clustering = rb_clust_users[rb_clust_users['users_type'] == 'experienced']

    ba_experts_threshold = ba_experts_threshold[columns]
    rb_experts_threshold = rb_experts_threshold[columns]
    ba_experts_clustering = ba_experts_clustering[columns]
    rb_experts_clustering = rb_experts_clustering[columns]

    fig = plot_scatter_active_time_total_reviews(ba_experts_threshold, rb_experts_threshold, ba_clust_users, rb_clust_users)
    figs.append(fig)
    
    return figs, ba_clust_summary, rb_clust_summary

