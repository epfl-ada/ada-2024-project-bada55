import argparse
from src.data.datasets import *

DATA_FOLDER = "data/"
GENERATED_FOLDER = "generated/"

def main(args):
    ba_dataset = BeerAdvocateDataset(data_folder=DATA_FOLDER, generated_folder=GENERATED_FOLDER, from_raw=True)
    rb_dataset = RateBeerDataset(data_folder=DATA_FOLDER, generated_folder=GENERATED_FOLDER, from_raw=True)
    ba_data = rb_dataset.get_data()
    rb_data = ba_dataset.get_data()

    

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Run model with optional cross validation"
    )

    args = parser.parse_args()

    main(args)