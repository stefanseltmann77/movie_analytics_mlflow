import logging

from movie_analytics.dataloader import DataLoader

loader = DataLoader()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader.update_from_s3('s3_ml_storage', 'stese-ml-datasets')
