import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from scipy import stats
from src.visualization.experts_selection_viz import *

def fig_percentage_experts_site(reviews: pd.DataFrame):
    
    users = reviews.groupby('user_id').agg(num_reviews=('text', 'count'))
    total_reviews = users['num_reviews'].sum()
    users['user_proportion_percentage'] = 1 / len(users) * 100
    users['review_proportion_percentage'] = users['num_reviews'] / total_reviews * 100
    users = users.sort_values(by= 'num_reviews', ascending= False).reset_index()
    users['cumulative_review_proportion'] = users['review_proportion_percentage'].cumsum()
    users['cumulative_user_proportion'] = users['user_proportion_percentage'].cumsum()

    last_experts_index = users[users['cumulative_review_proportion'] < 50].index[-1] + 1
    last_10_pourcentage_index = users[users['cumulative_user_proportion'] < 10].index[-1] + 1
    last_50_pourcentage_index = users[users['cumulative_user_proportion'] < 50].index[-1] + 1
        
    review_proportion_experts = users.loc[last_experts_index]['cumulative_review_proportion']
    user_proportion_experts = users.loc[last_experts_index]['cumulative_user_proportion']

    users['cumulative_review_proportion'] += -review_proportion_experts
    users['cumulative_user_proportion'] += -user_proportion_experts
    review_proportion_10 = users.loc[last_10_pourcentage_index]['cumulative_review_proportion']
    user_proportion_10 = users.loc[last_10_pourcentage_index]['cumulative_user_proportion']

    users['cumulative_review_proportion'] += -review_proportion_10
    users['cumulative_user_proportion'] += -user_proportion_10
    review_proportion_50 = users.loc[last_50_pourcentage_index]['cumulative_review_proportion']
    user_proportion_50 = users.loc[last_50_pourcentage_index]['cumulative_user_proportion']

    users['cumulative_review_proportion'] += -review_proportion_50
    users['cumulative_user_proportion'] += -user_proportion_50
    review_proportion_less_active = users.iloc[-1]['cumulative_review_proportion']
    user_proportion_less_active = users.iloc[-1]['cumulative_user_proportion']

    pourcentage_users = [user_proportion_experts, user_proportion_10, user_proportion_50, user_proportion_less_active]
    pourcentage_reviews = [review_proportion_experts, review_proportion_10, review_proportion_50, review_proportion_less_active]
    data = np.array([pourcentage_users, pourcentage_reviews])
    return data / 100

def fig_experts_selection(ba_reviews: pd.DataFrame, rb_reviews: pd.DataFrame):
    data_exp_ba = fig_percentage_experts_site(ba_reviews)
    data_exp_rb = fig_percentage_experts_site(rb_reviews)
    return fig_exp_site(data_exp_ba, data_exp_rb)