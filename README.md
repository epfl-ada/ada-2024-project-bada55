# BeerAdvocate vs. RateBeer: A Comparative Study of Online Beer Communities

## Abstract
At the age of twelve some people have to sign up for a music platform account. And at the age of eighteen some are faced with sign up for a beer platform account. This choice, motivated by personal reasons, leads to some disparities between our two main characters, BeerAdvocate and RateBeer. This project aims to examine the unique characteristics of each platform by identifying trends in user demographics, beer ratings, and style preferences. We will highlight key tendencies and influential factors that differentiate the communities, ultimately offering insights for new subscribe(e)rs, researchers, and industry professionals. Through this comparative analysis, readers will gain a comprehensive overview of the relevant dynamics that shape beer rating and consumer tastes online.

## Quickstart

Please follow the usage to run the project:
```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n bada55 python=3.11 or ...
conda activate bada55

# install requirements
pip install -r pip_requirements.txt

# run process.ipynb once before running the results notebook (results.ipynb) to preprocess the data

# run results.ipynb, which contains all the results 
```

## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│   ├── BeerAdvocate                <- BeerAdvocate data files
│   ├── matched_beer_data           <- matched_beer data files
│   ├── RateBeer                    <- BeerAdvocate data files
│   ├── zipped                      <- Original zipped project data files
│   ├── idm2012.pdf                 <- Learning Attitudes and Attributes from Multi-Aspect Reviews
│   ├── Lederrey-West_WWW-18.pdf    <- Measuring Herding Effects in Product Ratings with Natural Experiments
│
├── generated                    <- Project processed data files
│   ├── ..                          <- new_.csv files & new_.parquet files
│   ├── ba_chunks                   <- Chunks to create new_ba_reviews
│   ├── rb_chunks                   <- Chunks to create new_rb_reviews
│   ├── figures                     <- Project figures
│
├── src                         <- Source code
│   ├── data                        <- Data directory
│   │   ├── process.ipynb               <- Generate usable data 
│   ├── models                      <- Model directory
│   ├── utils                       <- Utility directory
│   ├── scripts                     <- Shell scripts
│
├── tests                       <- Tests of any kind
│   ├── exploring.ipynb             <- More codes/visualizations which will be useful for P3
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

## Project tasks and methodology

### Research questions
We decided to group the different subquestions that were linked into different “tasks”. These are the main questions but we might go deeper in some of them depending on what we find interesting. 

#### Task 1 : Number of users
“Which platform attracts more reviewers over time ?”
    a. How does the number of reviewers evolve over time on both platforms ?
    b. How are each platform’s reviewers distributed around the world over time ? Which platform is dominant in each region ?

#### Task 2 : Number of reviews
“On which platform is there the most activity in terms of reviews ?”
    a. How does the number of reviews evolve over time on both platforms ?
    b. How are each platform’s activity distributed around the world over time ? 

#### Task 3 : User activity
“On which platform are the reviewers most active ?”
    How is reviewers’ activity shared out ? What proportion of users have written more than a certain amount of reviews ?

#### Task 4 : General trends
“What are the top-rated beers, styles, and breweries by year ? And overall ?”

#### Task 5 : User rating
“How do reviewers rate the main types of beer on average on each platform ?”

### Methods 
At this stage, the work has been organized into three notebooks:
one covering the pre-processing of the data (preprocess.ipynb), where both the dataset BeerAdvocate and RateBeer were cleaned and prepared for specific tasks
another for the research that will help us directly to answer the tasks (results.ipynb) 
a final one covering research that doesn't directly answer our questions, but which has nevertheless helped us to understand the data (exploration.ipynb) and can maybe help us later for some insights.

#### Task 1: Number of Users
    a. Extract the user's first activity date. Aggregate the data by year to count unique users joining each year on each platform. 
    b. Use the location, to categorize users by region and year. Group users by region and year, then calculate which platform has more users in each region.

#### Task : 2 Number of Reviews
    a. Aggregate the review count by year for each platform. 
    b. Link each review with its author’s location. Group reviews by location and year to analyze regional activity levels. For additional insight, normalize the review count by the number of users in each region.

#### Task 3: User Activity
Extract the number of reviews for each user. Calculate the proportion of users with over a certain amount of reviews, as a percentage of total users on each platform.

#### Task 4: General Trends
Identify top-rated beers, styles, and breweries by filtering records with high average scores and reviewing frequency. Analyze trends by year and overall for each platform. Aggregation by beer style and brewery can highlight popular preferences and consistent high ratings over time.

#### Task 5: User Ratings
Identify the main beer types (categorize each beer in a type). Calculate the average rating for each beer type on both platforms. We will have to be careful on interpreting this because each platform has a different way of evaluating the beers and it can also depend on several other factors (like the number of reviews for a type etc.)

Some ideas for the visualization
For the questions of evolution over time, line charts could be interesting. For the reviewers distribution, we can use a world map with color-coded countries to show the dominant platform per region, as for the US elections. To present the proportion of active users, we could do a histogram with a vertical line showing the threshold, which can be modified with a cursor. For the general trends, the visualizations could include bar charts or ranked lists for a comparative view of each platform’s most popular choices.

### Timeline 
Week 1 : Individual task work
Week 2 : Individual task work (workload rebalancing if necessary) with visualization part
29.11 : Homework H2 deadline
Week 3: Tasks completion (workload rebalancing if necessary) and getting started with the website
Week 4 : Website formatting and datastory writing
Week 5: Datastory finishing
20.12 : Project milestone P3 deadline

### Distribution of tasks
François GOYBET : task 5
Zacharie LIBAUD : task 2
Zoé MONNARD : task 1
William SCHMID : task 3
Pierre TESSIER : task 4

### References
Website used for the classification of the beers : https://minuman.com/blogs/all/different-types-of-beer?srsltid=AfmBOorpJOU4ON1JA8B0A-XSaYkkZPYItTwABj1A9FSaKL7SeX9V8BAB