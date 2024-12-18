import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots

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
                labels={'Coeff': 'Percentage (%)'},  
                title="Comparison of coefficients by topic (in %)")

    fig.update_layout(
        yaxis=dict(tickformat='.0%', range=[0, 1]),  
        barmode='stack',  
        xaxis_title='Site',
        yaxis_title='Percentage (%)',
        template='plotly_white',
    )

    fig.update_traces(texttemplate='%{text}', textposition='inside')
    
    fig.update_layout(
        height=600,  
        width=800    
    )

    return fig

def fig_pred_vs_reel(
        ba_reviews: pd.DataFrame,
        rb_reviews: pd.DataFrame,
        predict_ba: callable,
        predict_rb: callable,
        r2_ba: float,
        r2_rb: float,
) -> go.Figure:
    y_ba = ba_reviews[["appearance","aroma","palate","taste","overall","rating"]]
    y_ba = (y_ba - y_ba.min())/(y_ba.max() - y_ba.min())

    y_rb = rb_reviews[["appearance","aroma","palate","taste","overall","rating"]]
    y_rb = (y_rb - y_rb.min())/(y_rb.max() - y_rb.min())

    y_real_ba = y_ba["rating"]
    y_pred_ba = predict_ba(y_ba[['appearance', 'aroma', 'palate', 'taste', 'overall']])

    y_real_rb = y_rb["rating"]
    y_pred_rb = predict_rb(y_rb[['appearance', 'aroma', 'palate', 'taste', 'overall']])

    range_viz = np.arange(len(y_real_ba))
    np.random.shuffle(range_viz)
    range_viz = range_viz[:int(0.001* len(y_real_ba))]
    y_real_ba = y_real_ba.values[range_viz]
    y_pred_ba = y_pred_ba.values[range_viz]

    range_viz = np.arange(len(y_real_rb))
    np.random.shuffle(range_viz)
    range_viz = range_viz[:int(0.001* len(y_real_rb))]
    y_real_rb = y_real_rb.values[range_viz]
    y_pred_rb = y_pred_rb.values[range_viz]

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=[f"BeerAdvocate R² = {r2_ba:.2f}", f"RateBeer R² = {r2_rb:.2f}"]
    )

    x_range_1 = np.linspace(min(y_real_ba.min(), y_pred_ba.min()), max(y_real_ba.max(), y_pred_ba.max()), 100)
    fig.add_trace(go.Scatter(
        x=y_real_ba, 
        y=y_pred_ba, 
        mode='markers', 
        name='Prediction ratings BeerAdvocate',
        marker=dict(color='blue')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_range_1, 
        y=x_range_1, 
        mode='lines', 
        name='y = x',
        line=dict(color='red', dash='dot')
    ), row=1, col=1)

    x_range_2 = np.linspace(min(y_real_rb.min(), y_pred_rb.min()), max(y_real_rb.max(), y_pred_rb.max()), 100)
    fig.add_trace(go.Scatter(
        x=y_real_rb, 
        y=y_pred_rb, 
        mode='markers', 
        name='Prediction ratings RateBeer',
        marker=dict(color='green')
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_range_2, 
        y=x_range_2, 
        mode='lines', 
        name='y = x',
        line=dict(color='red', dash='dot')
    ), row=1, col=2)

    fig.update_layout(
        height=600,
        width=1200,
        showlegend=True,  
        legend=dict(x=0.5, y=-0.2, xanchor='center', orientation='h')  
    )

    fig.update_xaxes(title_text="Real Rating", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Rating", row=1, col=1)

    fig.update_xaxes(title_text="Real Rating", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Rating", row=1, col=2)

    return fig