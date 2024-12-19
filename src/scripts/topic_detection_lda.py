print("ezoignoi")
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import PlaintextCorpusReader
from gensim.models import LdaMulticore
import spacy
from src.visualization.topic_detection_lda_viz import *
from wordcloud import STOPWORDS

NUM_TOPIC = 10
LIMIT = 10000 # how many chunks total
SIZE = 50 # how many sentences per chunk/page
TARGET_TOTAL_SENTENCES = LIMIT * SIZE # how many sentences total

def generate_total_reviews(reviews_experts_en: pd.dataFrame, dataset_name : str):
    df_current_total_reviews = pd.DataFrame(columns= reviews_experts_en.columns)
    max_iter = max(reviews_experts_en.groupby('user_id').agg(number_reviews= 
                                                    ('text', 'count')).reset_index().sort_values(by= 'number_reviews', ascending= False).number_reviews)
    current_total_sentences = 0
    if dataset_name == "BeerAdovocate":
        iteration = 13 * 5 #as 13 iter represent approximately 100_000 sentences for BeerAdvocateExperts
    elif dataset_name == "RateBeer":
        iteration = 32 * 5 #as 32 iter represent approximately 100_000 sentences for RateBeerExperts
    else:
        print("Wrong dataset name")
        return
        
    while current_total_sentences < TARGET_TOTAL_SENTENCES:
        for user in reviews_experts_en['user_id'].unique():
            df_current_total_reviews = pd.concat([df_current_total_reviews, 
                                                    reviews_experts_en[reviews_experts_en['user_id'] == user].head(iteration)])
        current_total_reviews = " ".join(review for review in df_current_total_reviews.text)
        current_total_sentences = len(sent_tokenize(current_total_reviews))
        print(f"Iteration n°{iteration}")
        print(f"BA current total sentences: {current_total_sentences}")
        df_current_total_reviews = pd.DataFrame(columns= reviews_experts_en.columns)
        if iteration > max_iter:
            print("Not enough sentences to reach the TARGET_TOTAL_SENTENCES")
            break
        iteration+= 1
    if dataset_name == "BeerAdovocate":
        output_file = "generated/corpus/ba_reviews_experts.txt"
    elif dataset_name == "RateBeer":
        output_file = "generated/corpus/rb_reviews_experts.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(current_total_reviews)
    print(f"saving file : '{output_file}'.")
    return

def get_chunks(l, n): #utils
    """Yield successive n-SIZEd chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def apply_LDA():

    nlp = spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    stopwords = set(STOPWORDS)
    stopwords.update(spacy_stopwords)
    stopwords.update(['beer', 'beers'])
    NUM_TOPIC = 10

    # Let's load our corpus via NLTK this time
    our_datasets = PlaintextCorpusReader('generated/corpus/', '.*.txt')
    print(our_datasets.fileids())
    
    # Get the chunks again (into smaller chunks)
    dataset_id = {f:n for n,f in enumerate(our_datasets.fileids())} # dictionary of books
    chunks = list()
    chunk_class = list() # this list contains the original dataset of the chunk, for evaluation
    
    for f in our_datasets.fileids():
        sentences = our_datasets.sents(f)
        print(f)
        print('Number of sentences:',len(sentences))
        # create chunks
        chunks_of_sents = [x for x in get_chunks(sentences,SIZE)] # this is a list of lists of sentences, which are a list of tokens
        chs = list()
        
        # regroup so to have a list of chunks which are strings
        for c in chunks_of_sents:
            grouped_chunk = list()
            for s in c:
                grouped_chunk.extend(s)
            chs.append(" ".join(grouped_chunk))
        print("Number of chunks:",len(chs),'\n')
        
        # filter to the LIMIT, to have the same number of chunks per book
        chunks.extend(chs[:LIMIT])
        chunk_class.extend([dataset_id[f] for _ in range(len(chs[:LIMIT]))])

    processed_docs = list()



    for doc in nlp.pipe(chunks, n_process=5, batch_SIZE=10):
    
        # Process document using Spacy NLP pipeline.
        ents = doc.ents  # Named entities
    
        # Keep only words (no numbers, no punctuation).
        # Lemmatize tokens, remove punctuation and remove stopwords.
        doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    
        # Remove common words from a stopword list and keep only words of length 3 or more.
        doc = [token for token in doc if token not in stopwords and len(token) > 2]
    
        # Add named entities, but only if they are a compound of more than word.
        doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
        processed_docs.append(doc)
    docs = processed_docs
    del processed_docs

    # Add bigrams too
    from gensim.models.phrases import Phrases
    
    # Add bigrams to docs (only ones that appear 15 times or more).
    bigram = Phrases(docs, min_count=15)
    
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # Create a dictionary representation of the documents, and filter out frequent and rare words.
    from gensim.corpora import Dictionary
    dictionary = Dictionary(docs)
    
    # Remove rare and common tokens.
    # Filter out words that occur too frequently or too rarely.
    max_freq = 0.5
    min_wordcount = 5
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
    
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    #MmCorpus.serialize("models/corpus.mm", corpus)
    
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of chunks: %d' % len(corpus))

    # models
    params = {'passes': 10, 'random_state': 42}
    base_models = dict()
    model = LdaMulticore(corpus=corpus, num_topics= NUM_TOPIC, id2word=dictionary, workers=6,
                    passes=params['passes'], random_state=params['random_state'])

    # Assign dominant topic to each chunk
    dominant_topics = []
    for i, chunk in enumerate(corpus):
        topic_distribution = model[chunk]  # Get topic probabilities
        dominant_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append(dominant_topic)
    
    # Combine results into a DataFrame
    df_topics = pd.DataFrame({
        'Chunk': range(len(corpus)),
        'Dataset': chunk_class,  # 0 = BeerAdvocate, 1 = RateBeer
        'Dominant_Topic': dominant_topics
    })
    
    # Map dataset names for readability
    df_topics['Dataset'] = df_topics['Dataset'].map({0: 'BeerAdvocate', 1: 'RateBeer'})

    # Group by dataset and dominant topic
    topic_distribution = df_topics.groupby(['Dataset', 'Dominant_Topic']).SIZE().unstack(fill_value=0)
    
    print("Topic Distribution by Dataset:")
    print(topic_distribution)

    # Display the top words for each topic
    for topic_id in range(NUM_TOPIC):
        print(f"Topic {topic_id}:")
        print(model.show_topic(topic_id, topn=10))
    
    fig_heatmap_topic_distribution = fig_heatmap_topic_distribution_dataset(topic_distribution)

    return fig_heatmap_topic_distribution