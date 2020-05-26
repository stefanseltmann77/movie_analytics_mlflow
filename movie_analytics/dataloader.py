import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from data.listings import oscar_winners, top_250_engl, worst_100


class DataLoader:
    """Loader class for the prepared profiler of movie metadata

    It automatically conducts the most basic preparation steps regardless which model the data
    is used for.
    """
    logger: logging.Logger

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_data(self, datafile: Path) -> DataFrame:
        self.logger.info(f"Loaded data from {datafile}")
        df = pd.read_csv(datafile,
                         index_col='imdb_movie_id',
                         error_bad_lines=False,
                         dtype={'rating_imdb_count': np.int64})
        return df

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
        df['is_oscar_winner'] = np.where(df.index.isin(oscar_winners), 1, 0)
        df['is_top250'] = np.where(df.index.isin(top_250_engl), 1, 0)
        df['is_worst100'] = np.where(df.index.isin(worst_100), 1, 0)
        return df

    def load_prepared_data(self, datafile: Path) -> DataFrame:
        df = self.load_data(datafile)
        df = self.clean_data(df)
        df = self.add_ranking_flags(df)
        return df
