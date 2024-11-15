
# Your project name
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.



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
│
├── src                         <- Source code
│   ├── data                        <- Data directory
│   │   ├── process.ipynb               <- Generate usable data 
│   ├── models                      <- Model directory
│   ├── utils                       <- Utility directory
│   ├── scripts                     <- Shell scripts
│
├── tests                       <- Tests of any kind
│   ├── exploring.ipynb             <- More useful codes/visualizations which will be useful for P3
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```