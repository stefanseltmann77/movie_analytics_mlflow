# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from oscar_analytics import OscarAnalytics

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 999)


class SomeModel:
    @staticmethod
    def predict(data: pd.DataFrame):
        oa = OscarAnalytics(Path("./data/dataset/movie_prof.csv"))
        df_movies = oa.load_data()
        df_movies = oa.clean_data(df_movies)
        df_movies = oa.add_ranking_flags(df_movies)
        df_movies.dropna(inplace=True)
        model = DecisionTreeClassifier(max_depth=3).fit(X=df_movies[['year', 'runtime']], y=df_movies.is_top250)
        return model.predict_proba(data)


def _load_pyfunc(path: str):
    return SomeModel()


# example_call
# curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns": ["year", "runtime"], "data":[[2012, 120]]}' http://127.0.0.1:5001/invocations
