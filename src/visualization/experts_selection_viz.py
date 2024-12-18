import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def fig_percentage_experts_plotly(data: np.array, name: str) -> go.Figure:

    categories = ['Expert', 'Intermediate', 'Novice', 'Debutant']
    bars = ['Users (%)', 'Reviews (%)']


    df_combined = pd.DataFrame(data, columns=categories) 
    df_combined['site'] = bars  
    
    df_combined = df_combined.melt(id_vars=['site'], var_name='index', value_name='Coeff')

    fig = px.bar(df_combined,
                 x="site", 
                 y="Coeff", 
                 color="index",  
                 text=df_combined["Coeff"].apply(lambda x: f"{x*100:.1f}%"), 
                 labels={'Coeff': 'Percentage (%)'},  
                 title=f"Comparison of Coefficients by Category in {name} (in %)",
                 )

    fig.update_layout(
        yaxis=dict(tickformat='.0%', range=[0, 1]),  
        barmode='stack',  
        xaxis_title='Category',
        yaxis_title='Percentage (%)',
        template='plotly_white'
    )

    fig.update_traces(texttemplate='%{text}', textposition='inside')

    fig.update_layout(
        height=600,  
        width=800    
    )

    return fig