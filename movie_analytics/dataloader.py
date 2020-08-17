import logging
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from pandas import DataFrame

from data.listings import OSCAR_WINNERS, TOP_250_ENGL, WORST_100
from movie_analytics.conf import project_dir


class DataLoader:
    """Loader class for the prepared profiler of movie metadata

    It automatically conducts the most basic preparation steps regardless
    which model the data is used for.
    """
    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_data(self, datafile: Path) -> DataFrame:
        """Loading the movie profiler containing the metadata on rowlevel"""
        self.logger.info(f"Loaded data from {datafile}")
        df = pd.read_parquet(datafile)
        df.set_index('imdb_movie_id', inplace=True)
        return df

    def load_manual_clustering(self, clusterfile: Path) -> DataFrame:
        """Loading a self set manual clustering of movies"""
        clusters = pd.read_parquet(clusterfile)
        clusters.set_index('imdb_movie_id', drop=True, inplace=True)
        return clusters

    def clean_data(self, df: DataFrame) -> DataFrame:
        self.logger.info(f"Cleaning Data")
        df.storyline = df.storyline.astype(str)
        df.synopsis = df.synopsis.astype(str)
        df.loc[df.synopsis.str.startswith("It looks like we don't have a Synopsis"), 'synopsis'] = np.NaN
        # take only cases with sensible size
        return df.loc[df.storyline.str.len() > 100]

    def add_ranking_flags(self, df: DataFrame) -> DataFrame:
        """Adding flags if record is part of predefined rankings"""
        self.logger.info("Adding potential target flags")
        df['is_oscar_winner'] = np.where(df.index.isin(OSCAR_WINNERS), 1, 0)
        df['is_top250'] = np.where(df.index.isin(TOP_250_ENGL), 1, 0)
        df['is_worst100'] = np.where(df.index.isin(WORST_100), 1, 0)
        return df

    def load_prepared_data(self, datafile: Path) -> DataFrame:
        df = self.load_data(datafile)
        df = self.clean_data(df)
        df = self.add_ranking_flags(df)
        return df

    def update_from_s3(self, profile_name: str, bucket_name: str):
        session = boto3.Session(profile_name=profile_name)
        s3 = session.resource('s3')
        buck = s3.Bucket(bucket_name)
        buck.download_file('clustering.prq', str(project_dir / Path('data', 'clustering.prq')))
        buck.download_file('profiler.prq', str(project_dir / Path('data', 'profiler.prq')))
        self.logger.info("Datasets updated.")
