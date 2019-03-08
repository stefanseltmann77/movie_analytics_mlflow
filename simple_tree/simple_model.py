# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
from pandas import np
from listings import *
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 999)

class OscarAnalytics:
    logger: logging.Logger
    datafile: Path

    def __init__(self, datafile: Path):
        self.logger = logging.getLogger(__name__)
        self.datafile = datafile

    def load_data(self) -> pd.DataFrame:
        self.logger.info(f"Loaded data from {self.datafile}")
        df = pd.read_csv(self.datafile, index_col='imdb_movie_id',
                         dtype={'rating_imdb_count': np.int64})
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Cleaning Data")
        df.storyline = df.storyline.astype(str)
        df.synopsis = df.synopsis.astype(str)
        df.loc[df.synopsis.str.startswith("It looks like we don't have a Synopsis"), 'synopsis'] = np.NaN
        # take only cases with sensible size
        return df.loc[df.storyline.str.len() > 100]

    def add_ranking_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Adding potential target flags")
        df['is_oscar_winner'] = np.where(df.index.isin(oscar_winners), 1, 0)
        df['is_top250'] = np.where(df.index.isin(top_250_engl), 1, 0)
        df['is_worst100'] = np.where(df.index.isin(worst_100), 1, 0)
        return df


def predict(data: pd.DataFrame):
    oa = OscarAnalytics(Path("movie_prof.csv"))
    df_movies = oa.load_data()
    df_movies = oa.clean_data(df_movies)
    df_movies = oa.add_ranking_flags(df_movies)
    df_movies.dropna(inplace=True)
    model = DecisionTreeClassifier(max_depth=3).fit(X=df_movies[['year', 'runtime']], y=df_movies.is_top250)
    return model.predict_proba(data)


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    # oa = OscarAnalytics(Path("./dataset/movie_prof.csv"))
    # df_movies = oa.load_data()
    # df_movies = oa.clean_data(df_movies)
    # df_movies = oa.add_ranking_flags(df_movies)
    # df_movies.dropna(inplace=True)
    # model = DecisionTreeClassifier(max_depth=3).fit(X=df_movies[['year', 'runtime']], y=df_movies.is_top250)
    # mlflow.sklearn.save_model(model, "./models/simple_tree")


