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

            self.rb_breweries.to_csv(generated_folder + RB_BEERS_DATASET)
            self.rb_beers.to_csv(generated_folder + RB_BREWERIES_DATASET)
            self.rb_users.to_csv(generated_folder + RB_USERS_DATASET)
            self.rb_reviews.to_parquet(generated_folder + "RateBeer/reviews.parquet")
            return
        
        self.rb_beers = pd.read_csv(generated_folder + RB_BEERS_DATASET)
        self.rb_breweries = pd.read_csv(generated_folder + RB_BREWERIES_DATASET)
        self.rb_users = pd.read_csv(generated_folder + RB_USERS_DATASET)
        self.rb_reviews = pd.read_parquet(generated_folder + "RateBeer/reviews.parquet")   

    def preprocess(self) -> None:

        # Breweries
        self.rb_breweries['country'] = self.rb_breweries['location'].apply(lambda name : name_to_country(name))
        self.rb_breweries['continent'] = self.rb_breweries['country'].apply(lambda country : country_continent_map.get(country, 'Unknown'))
        
        # Beers
        rb_dict_id_br_concat = dict(zip(self.rb_breweries['id'], self.rb_breweries['continent']))
        self.rb_beers['continent'] = self.rb_beers['brewery_id'].apply(lambda id_: rb_dict_id_br_concat.get(id_))
        self.rb_beers['type'] = self.rb_beers['style'].apply(lambda style: style_to_type)

        # Users
        self.rb_users['location'] = self.rb_users['location'].astype(str)
        self.rb_users['country'] = self.rb_users['location'].apply(lambda name : name_to_country(name))
        self.rb_users['continent'] = self.rb_users['country'].apply(lambda country : country_continent_map.get(country, 'Unknown'))
        self.rb_users['joined'] = pd.to_datetime(self.rb_users['joined'], unit='s').dt.strftime('%d/%m/%Y')

        # Reviews
        cols_to_numeric = ['beer_id', 'brewery_id', 'abv', 'date', 'appearance', 'aroma', 'palate', 'taste', 'overall', 'rating']
        self.rb_reviews[cols_to_numeric] = self.rb_reviews[cols_to_numeric].apply(pd.to_numeric, errors = 'coerce')
        self.rb_reviews['date'] = pd.to_datetime(self.rb_reviews['date'], unit='s').dt.strftime('%d/%m/%Y')
        self.rb_reviews['continent'] = self.rb_reviews['brewery_id'].apply(lambda id_: rb_dict_id_br_concat.get(int(id_)))
        self.rb_reviews['type'] = self.rb_reviews['style'].apply(style_to_type)

    def read_rb_reviews(self, data_folder: str = "data/", chunk_size: int = 1_000_000) -> pd.DataFrame:

        columns = ['beer_name', 'beer_id', 'brewery_name', 'brewery_id', 'style', 'abv', 'date', 
                'user_name', 'user_id', 'appearance', 'aroma', 'palate', 'taste', 'overall', 
                'rating']
        data = []
        entry_count = 0
        chunk_count = 0
        current_entry = {}

        with open(data_folder+RB_REVIEWS_DATASET, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    if line.startswith('text:'):
                        continue
                    if ':' in line:
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
        }

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

            self.ba_breweries.to_csv(generated_folder + BA_BEERS_DATASET)
            self.ba_beers.to_csv(generated_folder + BA_BREWERIES_DATASET)
            self.ba_users.to_csv(generated_folder + BA_USERS_DATASET)
            self.ba_reviews.to_parquet(generated_folder + "BeerAdvocate/reviews.parquet")
            return
        
        self.ba_beers = pd.read_csv(generated_folder + BA_BEERS_DATASET)
        self.ba_breweries = pd.read_csv(generated_folder + BA_BREWERIES_DATASET)
        self.ba_users = pd.read_csv(generated_folder + BA_USERS_DATASET)
        self.ba_reviews = pd.read_parquet(generated_folder + "BeerAdvocate/reviews.parquet")   

    def preprocess(self) -> None:

        # Breweries
        self.ba_breweries['country'] = self.ba_breweries['location'].apply(lambda name : name_to_country(name))
        self.ba_breweries['continent'] = self.ba_breweries['country'].apply(lambda country : country_continent_map.get(country, 'Unknown'))
        
        # Beers
        rb_dict_id_br_concat = dict(zip(self.ba_breweries['id'], self.ba_breweries['continent']))
        self.ba_beers['continent'] = self.ba_beers['brewery_id'].apply(lambda id_: rb_dict_id_br_concat.get(id_))
        self.ba_beers['type'] = self.ba_beers['style'].apply(lambda style: style_to_type)

        # Users
        self.ba_users['location'] = self.ba_users['location'].astype(str)
        self.ba_users['country'] = self.ba_users['location'].apply(lambda name : name_to_country(name))
        self.ba_users['continent'] = self.ba_users['country'].apply(lambda country : country_continent_map.get(country, 'Unknown'))
        self.ba_users['joined'] = pd.to_datetime(self.ba_users['joined'], unit='s').dt.strftime('%d/%m/%Y')

        # Reviews
        cols_to_numeric = ['beer_id', 'brewery_id', 'abv', 'date', 'appearance', 'aroma', 'palate', 'taste', 'overall', 'rating']
        self.ba_reviews[cols_to_numeric] = self.ba_reviews[cols_to_numeric].apply(pd.to_numeric, errors = 'coerce')
        self.ba_reviews['date'] = pd.to_datetime(self.ba_reviews['date'], unit='s').dt.strftime('%d/%m/%Y')
        self.ba_reviews['continent'] = self.ba_reviews['brewery_id'].apply(lambda id_: rb_dict_id_br_concat.get(int(id_)))
        self.ba_reviews['type'] = self.ba_reviews['style'].apply(style_to_type)

    def read_ba_reviews(self, data_folder: str = "data/", chunk_size: int = 1_000_000) -> pd.DataFrame:

        columns = ['beer_name', 'beer_id', 'brewery_name', 'brewery_id', 'style', 'abv', 'date', 
                'user_name', 'user_id', 'appearance', 'aroma', 'palate', 'taste', 'overall', 
                'rating']
        data = []
        entry_count = 0
        chunk_count = 0
        current_entry = {}

        with open(data_folder+BA_REVIEWS_DATASET, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    if line.startswith('text:'):
                        continue
                    if ':' in line:
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
        }