import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import spacy
from src.visualization.topic_detection_naive_viz import *

BEER_MASK = np.array(Image.open("data/beerCan.jpg"))
spacy.load('en_core_web_sm') #to check if it is here
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
stopwords = set(STOPWORDS)
stopwords.update(spacy_stopwords)
stopwords.update(['beer', 'beers'])

def generate_wordcloud(reviews_experts_en: pd.DataFrame) -> WordCloud:
    text = " ".join(review for review in reviews_experts_en['text'])
    text_lower = text.lower()
    del text
    wc = WordCloud(
        background_color="white", 
        mask=BEER_MASK,
        stopwords=stopwords,
        min_font_size=6
    )
    wc.generate(text_lower)
    del text_lower
    fig = fig_show_wordcloud(wc)
    return fig