import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def fig_exp_site(data_exp_ba: np.array, data_exp_rb: np.array) -> go.Figure:

    categories = ['Expert', 'Intermediate', 'Novice', 'Debutant']
    bars = ['Users (%)', 'Reviews (%)']
    
    data1_df = pd.DataFrame(data_exp_ba, columns=categories)
    data1_df['site'] = bars
    data1_df['dataset'] = 'BeerAdvocate'
    
    data2_df = pd.DataFrame(data_exp_rb, columns=categories)
    data2_df['site'] = bars
    data2_df['dataset'] = 'RateBeer'
    
    df_combined = pd.concat([data1_df, data2_df], ignore_index=True)
    
    df_melted = df_combined.melt(id_vars=['site', 'dataset'], var_name='index', value_name='Coeff')
    
    fig = px.bar(df_melted,
                 x="site", 
                 y="Coeff", 
                 color="index",  
                 text=df_melted["Coeff"].apply(lambda x: f"{x:.1f}%"), 
                 facet_col="dataset",  
                 labels={'Coeff': 'Percentage (%)', 'site': 'Category'},  
                 title="Comparison of Coefficients by Category for Both Datasets",
                 )
    
    fig.update_layout(
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        barmode='stack',  
        xaxis_title='Category',
        yaxis_title='Percentage (%)',
        template='plotly_white',
        height=600,
        width=1000
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='inside')
    return fig  