import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from scipy import stats
from src.visualization.notation_system_viz import *

def lin_reg_model(reviews: pd.DataFrame):

    rating_df = reviews[["appearance","aroma","palate","taste","overall","rating"]]
    rating_df = (rating_df - rating_df.min())/(rating_df.max() - rating_df.min())
    model = smf.ols(formula='rating ~ appearance + aroma + palate + taste + overall - 1', data=rating_df).fit()

    return model

def fig_notation_system(ba_reviews: pd.DataFrame, rb_reviews: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    model_ba = lin_reg_model(ba_reviews)
    model_rb = lin_reg_model(rb_reviews)

    fig_comp_coeff = fig_comp_coeff_topic(model_ba.params,model_rb.params)
    fig_pred_reel = fig_pred_vs_reel(ba_reviews, rb_reviews, model_ba.predict, model_rb.predict, model_ba.rsquared, model_rb.rsquared)
    return (fig_comp_coeff, fig_pred_reel)