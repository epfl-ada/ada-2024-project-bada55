import pandas as pd
import matplotlib.pyplot as plt

def fig_pie_proportion_languages(df_languages: pd.DataFrame, name_dataset: str) -> plt.Figure:

    num_english_reviews = df_languages.loc['__en', 'num_languages'] if '__en' in df_languages.index else 0
    total_reviews = df_languages['num_languages'].sum()
    num_other_reviews = total_reviews - num_english_reviews
    pie_data = {'English': num_english_reviews, 'Other languages': num_other_reviews}
    fig, ax = plt.subplots()
    ax.pie(
        pie_data.values(), 
        labels=pie_data.keys(), 
        autopct='%1.2f%%', 
        startangle=140, 
        colors=['#ff9999', '#66b3ff']
    )
    ax.set_title(f"Proportion of English Reviews vs Other Languages for {name_dataset} Experts")
    plt.close()
    return fig
