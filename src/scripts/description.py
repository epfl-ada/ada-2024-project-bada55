import pandas as pd

def reviews_caracteristics(reviews: pd.DataFrame, name_dataset: str) -> None :
  
  num_unique_users = reviews['user_id'].nunique()
  num_reviews = reviews['text'].count()
  num_unique_styles = reviews['style'].nunique()
  num_unique_breweries = reviews['brewery_name'].nunique()
  
  print("Basic descriptive analysis {name_dataset} : ")
  print(f"Number of unique users: {num_unique_users}")
  print(f"Number of reviews: {num_reviews}")
  print(f"Number of unique beer styles: {num_unique_styles}")
  print(f"Number of breweries: {num_unique_breweries}")