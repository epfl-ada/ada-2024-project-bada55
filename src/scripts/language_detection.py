import pandas as pd
import numpy as np
import fasttext
from src.visualization.language_detection_viz import *

def detect_language_fasttext(text: str, model):
    ''' 
    This function detects the language of a given text using a pre-trained FastText model.
    '''
    try:
        if pd.isnull(text) or text.strip() == "":
            return "Unknown"
        prediction = model.predict(text.strip())
        return prediction[0][0].replace("_label_", "") 
    except Exception as e:
        print(e)
        return "unknown"

def apply_language_detection(reviews_experts: pd.DataFrame, name_dataset: str):
    model = fasttext.load_model("data/bin/lid.176.bin")
    reviews_experts['language'] = reviews_experts['text'].apply(lambda text: detect_language_fasttext(text, model))
    df_languages = reviews_experts.groupby('language').agg(num_languages =('language', 'count'))
    total_language = df_languages['num_languages'].sum()
    df_languages['review_proportion_percentage'] = (df_languages['num_languages'] / total_language * 100).round(5)
    df_languages = df_languages.sort_values(by= 'num_languages', ascending= False)
    print("Languages detected for {name_dataset}: ")
    fig_pie_proportion = fig_pie_proportion_languages(df_languages, name_dataset)
    reviews_experts_en = reviews_experts[reviews_experts['language'] == '__en'].copy()
    return df_languages, fig_pie_proportion, reviews_experts_en

