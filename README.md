# Which Platform is Best Suited for You?

This is the repository of the <a href="https://zaclibaud.github.io/bADA55.github.io/" target="_blank" rel="noopener noreferrer">website</a> for the <a href="https://edu.epfl.ch/coursebook/fr/applied-data-analysis-CS-401" target="_blank" rel="noopener noreferrer">CS-401 Applied Data Analysis</a> EPFL course project of the team bADA55.

## Abstract

In a world where decisions are increasingly influenced by online platforms, choosing one beer review platform between BeerAdvocate or RateBeer, can feel like choosing between Spotify and Deezer. Just as music lovers want to know which streaming service better matches their taste, beer enthusiasts want to understand which platform offers more reliable and insightful reviews. Our goal is to help answer this question by diving deep into the unique characteristics of both platforms to compare them.

## Quickstart

Please follow the usage to run the project:
```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-bada55.git
cd ada-2024-project-bada55

# [OPTIONAL] create conda environment
conda create -n bada55 python=3.11 or ...
conda activate bada55

# install requirements
pip install -r requirements.txt

# run results.ipynb, which contains all the results 
#the two functions : BeerAdvocateDataset and RateBeerDataset are called at the beginning to preprocess both datasets
```

## Project Structure

The directory structure of new project looks like this:

<code>
│   .gitignore
│   main.py
│   pip_requirements.txt
│   README.md
│   result.ipynb
│
├───data
│   │   beerCan.jpg
│   ├───BeerAdvocate
│   │   └───reviews.txt
│   ├───bin
│   └───RateBeer
│       └───reviews.txt
│
├───generated
│   ├───BeerAdvocate
│   │   └───ba_chunks
│   │
│   ├───corpus
│   │
│   └───RateBeer
│       └───rb_chunks
│
└───src
    ├───data
    │       datasets.py
    │       process.ipynb
    │
    ├───scripts
    │       clustering.py
    │       description.py
    │       experts_selection.py
    │       language_detection.py
    │       notation_system.py
    │       sentiment_analysis.py
    │       topic_detection_lda.py
    │       topic_detection_naive.py
    │       _init_.py
    │
    ├───utils
    │       helpers.py
    │       _init_.py
    │
    └───visualization
            clustering_viz.py
            experts_selection_viz.py
            language_detection_viz.py
            notation_system_viz.py
            sentiment_analysis_viz.py
            topic_detection_lda_viz.py
            topic_detection_naive_viz.py
            _init_.py

</code>

## Research questions and according methods

### How does the rating work on both platform ?
The same overall rating on BeerAdvocate and RateBeer does not reflect identical scores across different topics. In theory, if a user provides the same scores and descriptions for a beer on both platforms, the overall score would still differ between the two sites.

**Method:** For this we will use a linear regression model to find the coefficient for each parameter (aroma, taste, palate, appearance, overall).

### Who are the interesting/experts users ?

Not all reviewers are equal, and some can be considered experts. But what makes an expert? By identifying patterns in user behavior, we aim to determine which users contribute reviews that are more detailed, consistent, and insightful. These “experts” are essential for understanding which platform fosters a community of knowledgeable reviewers.

**Method:** For this, our goal is to use k-means clustering with features like total reviews, mean time and standard deviation spacing, and rating std to see if we can extract a group of “experts” users. 
We will also use another method which sets a threshold on the number of reviews, the experts here are those who wrote most of the reviews.

### What language do they speak ?

Before going into the analysis of the reviews of the experts users, we identify their languages.

**Method:** using fasttext.

### Which sentiment do they express the most ?

Using SentimentIntensityAnalyzer from Vader we aim to see the sentiment scores and the overall sentiment of each review to compare both plateforms.

### What kind of words do they use ?

Once we have identified the experts we are interested in, we can analyse their vocabulary to compare the two platforms. 

**Method:** To do this, we will first use the WorldClouds method to get an initial overview before using the Latent Dirichlet Allocation (LDA) method for a more detailed analysis of the topics used.

## Distribution of tasks

- François GOYBET : linear regression for ratings, creating .py files, kmeans
- Zacharie LIBAUD : preprocessing, language analysis, website
- Zoé MONNARD : language detection, sentimental analysis, website
- William SCHMID : kmean clustering
- Pierre TESSIER : kmeans, website
