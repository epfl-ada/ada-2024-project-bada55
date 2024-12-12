import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def fig_comp_coeff_topic(
        coeff_ba: pd.Series,
        coeff_rb: pd.Series
        ) -> go.Figure:
    
    ba_coeff = pd.DataFrame({
        'Coeff': coeff_ba,
        'topic': ['appearance', 'aroma', 'palate', 'taste', 'overall'],
        'site': 'Beer Advocate'
    })

    rb_coeff = pd.DataFrame({
        'Coeff': coeff_rb,
        'Theme': ['appearance', 'aroma', 'palate', 'taste', 'overall'],
        'site': 'Rate Beer'
    })

    df_combined = pd.concat([ba_coeff, rb_coeff], axis=0).reset_index()
    fig = px.bar(df_combined,
                x="site", 
                y="Coeff", 
                color="index",  
                text=df_combined["Coeff"].apply(lambda x: f"{x*100:.1f}%"),  
                labels={'Coeff': 'Pourcentage (%)'},  
                title="Comparison of coefficients by topic (in %)")

    fig.update_layout(
        yaxis=dict(tickformat='.0%', range=[0, 1]),  
        barmode='stack',  
        xaxis_title='Site',
        yaxis_title='Pourcentage (%)',
        template='plotly_white',
    )

    fig.update_traces(texttemplate='%{text}', textposition='inside')

    return fig