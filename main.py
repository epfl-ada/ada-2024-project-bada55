import argparse
from src.data.datasets import *

DATA_FOLDER = "data/"
GENERATED_FOLDER = "generated/"

def main(args):
    rb_dataset = RateBeerDataset(data_folder=DATA_FOLDER, generated_folder=GENERATED_FOLDER, from_raw=False)
    ba_dataset = BeerAdvocateDataset(data_folder=DATA_FOLDER, generated_folder=GENERATED_FOLDER, from_raw=False)
    data = rb_dataset.get_data()
    data = ba_dataset.get_data()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Run model with optional cross validation"
    )

    args = parser.parse_args()

    main(args)