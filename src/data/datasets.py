import numpy as np
import pandas as pd
import glob
from src.utils import *

class RateBeerDataset():

    def __init__(
            self, 
            data_folder: str = "data/",
            generated_folder: str = "generated/",
            from_raw: bool = True,
            ) -> None:

        if from_raw:
            # Read Beers, Breweries and Users
            self.rb_beers = pd.read_csv(data_folder + RB_BEERS_DATASET)
            self.rb_breweries = pd.read_csv(data_folder + RB_BREWERIES_DATASET)
            self.rb_users = pd.read_csv(data_folder + RB_USERS_DATASET)
            
            # Reviews
            self.rb_reviews = self.read_rb_reviews(data_folder=data_folder, chunk_size=1_000_000)

            self.preprocess()
            
            # Experts 
            self.rb_experts = self.define_experts(threshold_experts_reviews=50)

            self.rb_breweries.to_csv(generated_folder + RB_BEERS_DATASET)
            self.rb_beers.to_csv(generated_folder + RB_BREWERIES_DATASET)
            self.rb_users.to_csv(generated_folder + RB_USERS_DATASET)
            self.rb_reviews.to_parquet(generated_folder + "RateBeer/reviews.parquet")
            self.rb_experts.to_parquet(generated_folder + "RateBeer/experts.parquet")
            return
        
        self.rb_beers = pd.read_csv(generated_folder + RB_BEERS_DATASET)
        self.rb_breweries = pd.read_csv(generated_folder + RB_BREWERIES_DATASET)
        self.rb_users = pd.read_csv(generated_folder + RB_USERS_DATASET)
        self.rb_reviews = pd.read_parquet(generated_folder + "RateBeer/reviews.parquet")  
        self.rb_experts = pd.read_parquet(generated_folder + "RateBeer/experts.parquet") 

    def preprocess(self) -> None:

        # Users
        rb_users_joined = self.rb_users.copy()
        rb_users_joined = rb_users_joined.drop(['nbr_ratings', 'user_name', 'location'], axis= 1)
        rb_users_joined = rb_users_joined.dropna()
        # remove duplicates
        rb_users_joined = rb_users_joined.drop_duplicates(subset='user_id', keep='first')  # keep the first occurrence of each duplicate
        rb_users_joined['joined'] = pd.to_datetime(rb_users_joined['joined'], unit='s')
        rb_users_joined['user_id'] = rb_users_joined['user_id'].astype(str)
        self.rb_users = rb_users_joined

        # Reviews 
        empty_text = self.rb_reviews['text'] == ''
        self.rb_reviews.drop(self.rb_reviews[empty_text].index, inplace= True)
        missing_values = np.where(pd.isnull(self.rb_reviews))

        if missing_values[0].size == 0:
            print("The dataset is clean, with no missing values.")
        else:
            print("The dataset has missing values at:")
            print("Row indices:", missing_values[0])
            print("Column indices:", missing_values[1])

        print(f"Shape before keeping ratings from users in users_joined: {self.rb_reviews.shape}")
        self.rb_reviews = self.rb_reviews[self.rb_reviews['user_id'].isin(rb_users_joined['user_id'])]
        print(f"Shape after keeping ratings from users in users_joined: {self.rb_reviews.shape}")

        cols_to_numeric = ['beer_id', 'brewery_id', 'abv', 'date', 'appearance', 'aroma', 'palate', 'taste', 'overall', 'rating']
        self.rb_reviews[cols_to_numeric] = self.rb_reviews[cols_to_numeric].apply(pd.to_numeric, errors = 'coerce')
        self.rb_reviews['date'] = pd.to_datetime(self.rb_reviews['date'], unit='s')

    def read_rb_reviews(self, data_folder: str = "data/", chunk_size: int = 1_000_000) -> pd.DataFrame:

        columns = ['beer_name', 'beer_id', 'brewery_name', 'brewery_id', 'style', 'abv', 'date', 
                'user_name', 'user_id', 'appearance', 'aroma', 'palate', 'taste', 'overall', 
                'rating', 'text']
        data = []
        entry_count = 0
        chunk_count = 0
        current_entry = {}

        with open(data_folder+RB_REVIEWS_DATASET, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        current_entry[key] = value
                else:
                    if current_entry:
                        data.append(current_entry)
                        current_entry = {}
                        entry_count += 1

                        # Save chunk when reaching chunk size
                        if entry_count >= chunk_size:
                            chunk_df = pd.DataFrame(data, columns=columns)
                            chunk_file_path = f"generated/RateBeer/rb_chunks/rb_reviews_chunk_{chunk_count}.parquet"
                            chunk_df.to_parquet(chunk_file_path)
                            print(f"Saved {chunk_file_path}")
                            data = []
                            entry_count = 0
                            chunk_count += 1
                            
        # Process any remaining entries after the loop
        if data:
            chunk_df = pd.DataFrame(data, columns=columns)
            chunk_file_path = f"generated/RateBeer/rb_chunks/rb_reviews_chunk_{chunk_count}.parquet"
            chunk_df.to_parquet(chunk_file_path)
            print(f"Saved {chunk_file_path}")

        rb_chunk_files = glob.glob("generated/RateBeer/rb_chunks/rb_reviews_chunk_*.parquet")
        rb_reviews = pd.concat([pd.read_parquet(rb_chunk) for rb_chunk in rb_chunk_files], ignore_index=True)
        return rb_reviews
    
    def get_data(self):
        return {
            "beers": self.rb_beers,
            "breweries": self.rb_breweries,
            "users": self.rb_users,
            "reviews": self.rb_reviews,
            "experts": self.rb_experts,
        }

    def define_experts(self, threshold_experts_reviews=50):
        rb_users = self.rb_reviews.copy().groupby('user_id').agg(num_reviews=('text', 'count'))
        rb_total_reviews = rb_users['num_reviews'].sum()
        rb_users['review_proportion_percentage'] = rb_users['num_reviews'] / rb_total_reviews * 100
        rb_users = rb_users.sort_values(by= 'num_reviews', ascending= False)

        rb_sum_review_proportion_experts = 0
        rb_experts_id = []
        for index, row in rb_users.iterrows():
            if rb_sum_review_proportion_experts >= threshold_experts_reviews:
                break
            rb_sum_review_proportion_experts += row['review_proportion_percentage']
            rb_experts_id.append(row.name)

        rb_reviews_experts = self.rb_reviews[self.rb_reviews['user_id'].isin(rb_experts_id)].copy()

        return rb_reviews_experts

        
class BeerAdvocateDataset():

    def __init__(
            self, 
            data_folder: str = "data/",
            generated_folder: str = "generated/",
            from_raw: bool = True,
            ) -> None:

        if from_raw:
            # Read Beers, Breweries and Users
            self.ba_beers = pd.read_csv(data_folder + BA_BEERS_DATASET)
            self.ba_breweries = pd.read_csv(data_folder + BA_BREWERIES_DATASET)
            self.ba_users = pd.read_csv(data_folder + BA_USERS_DATASET)
            
            # Reviews
            self.ba_reviews = self.read_ba_reviews(data_folder=data_folder, chunk_size=1_000_000)

            self.preprocess()

            # Experts 
            self.ba_experts = self.define_experts(threshold_experts_reviews=50)

            self.ba_breweries.to_csv(generated_folder + BA_BEERS_DATASET)
            self.ba_beers.to_csv(generated_folder + BA_BREWERIES_DATASET)
            self.ba_users.to_csv(generated_folder + BA_USERS_DATASET)
            self.ba_reviews.to_parquet(generated_folder + "BeerAdvocate/reviews.parquet")
            self.ba_experts.to_parquet(generated_folder + "BeerAdvocate/experts.parquet")
            return
        
        self.ba_beers = pd.read_csv(generated_folder + BA_BEERS_DATASET)
        self.ba_breweries = pd.read_csv(generated_folder + BA_BREWERIES_DATASET)
        self.ba_users = pd.read_csv(generated_folder + BA_USERS_DATASET)
        self.ba_reviews = pd.read_parquet(generated_folder + "BeerAdvocate/reviews.parquet")   
        self.ba_experts = pd.read_parquet(generated_folder + "BeerAdvocate/experts.parquet") 

    def preprocess(self) -> None:

        # Users
        ba_users_joined = self.ba_users.copy()
        ba_users_joined = ba_users_joined.drop(['nbr_ratings', 'user_name', 'location'], axis= 1)
        ba_users_joined = ba_users_joined.dropna()
        # remove duplicates
        ba_users_joined = ba_users_joined.drop_duplicates(subset='user_id', keep='first')  # keep the first occurrence of each duplicate
        ba_users_joined['joined'] = pd.to_datetime(ba_users_joined['joined'], unit='s')
        ba_users_joined['user_id'] = ba_users_joined['user_id'].astype(str)
        self.ba_users = ba_users_joined

        # Reviews 
        empty_text = self.ba_reviews['text'] == ''
        self.ba_reviews.drop(self.ba_reviews[empty_text].index, inplace= True)
        missing_values = np.where(pd.isnull(self.ba_reviews))

        if missing_values[0].size == 0:
            print("The dataset is clean, with no missing values.")
        else:
            print("The dataset has missing values at:")
            print("Row indices:", missing_values[0])
            print("Column indices:", missing_values[1])

        print(f"Shape before keeping ratings from users in users_joined: {self.ba_reviews.shape}")
        self.ba_reviews = self.ba_reviews[self.ba_reviews['user_id'].isin(ba_users_joined['user_id'])]
        print(f"Shape after keeping ratings from users in users_joined: {self.ba_reviews.shape}")

        cols_to_numeric = ['beer_id', 'brewery_id', 'abv', 'date', 'appearance', 'aroma', 'palate', 'taste', 'overall', 'rating']
        self.ba_reviews[cols_to_numeric] = self.ba_reviews[cols_to_numeric].apply(pd.to_numeric, errors = 'coerce')
        self.ba_reviews['date'] = pd.to_datetime(self.ba_reviews['date'], unit='s')

    def read_ba_reviews(self, data_folder: str = "data/", chunk_size: int = 1_000_000) -> pd.DataFrame:

        columns = ['beer_name', 'beer_id', 'brewery_name', 'brewery_id', 'style', 'abv', 'date', 
                'user_name', 'user_id', 'appearance', 'aroma', 'palate', 'taste', 'overall', 
                'rating', 'text']
        data = []
        entry_count = 0
        chunk_count = 0
        current_entry = {}

        with open(data_folder+BA_REVIEWS_DATASET, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        current_entry[key] = value
                else:
                    if current_entry:
                        data.append(current_entry)
                        current_entry = {}
                        entry_count += 1

                        # Save chunk when reaching chunk size
                        if entry_count >= chunk_size:
                            chunk_df = pd.DataFrame(data, columns=columns)
                            chunk_file_path = f"generated/BeerAdvocate/ba_chunks/ba_reviews_chunk_{chunk_count}.parquet"
                            chunk_df.to_parquet(chunk_file_path)
                            print(f"Saved {chunk_file_path}")
                            data = []
                            entry_count = 0
                            chunk_count += 1
                            
        # Process any remaining entries after the loop
        if data:
            chunk_df = pd.DataFrame(data, columns=columns)
            chunk_file_path = f"generated/BeerAdvocate/ba_chunks/ba_reviews_chunk_{chunk_count}.parquet"
            chunk_df.to_parquet(chunk_file_path)
            print(f"Saved {chunk_file_path}")

        ba_chunk_files = glob.glob("generated/BeerAdvocate/ba_chunks/ba_reviews_chunk_*.parquet")
        ba_reviews = pd.concat([pd.read_parquet(ba_chunk) for ba_chunk in ba_chunk_files], ignore_index=True)
        return ba_reviews
    
    def get_data(self):
        return {
            "beers": self.ba_beers,
            "breweries": self.ba_breweries,
            "users": self.ba_users,
            "reviews": self.ba_reviews,
            "experts": self.ba_experts,
        }
    
    def define_experts(self, threshold_experts_reviews=50):
        ba_users = self.ba_reviews.groupby('user_id').agg(num_reviews=('text', 'count'))
        ba_total_reviews = ba_users['num_reviews'].sum()
        ba_users['review_proportion_percentage'] = ba_users['num_reviews'] / ba_total_reviews * 100
        ba_users = ba_users.sort_values(by= 'num_reviews', ascending= False)

        ba_sum_review_proportion_experts = 0
        ba_experts_id = []
        for index, row in ba_users.iterrows():
            if ba_sum_review_proportion_experts >= threshold_experts_reviews:
                break
            ba_sum_review_proportion_experts += row['review_proportion_percentage']
            ba_experts_id.append(row.name)

        ba_reviews_experts = self.ba_reviews[self.ba_reviews['user_id'].isin(ba_experts_id)].copy()

        return ba_reviews_experts