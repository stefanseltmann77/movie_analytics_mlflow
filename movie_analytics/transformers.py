from typing import Sequence

import pandas as pd
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


class BagOfWords(TransformerMixin):
    pipe: Pipeline
    text_col_name: str

    def __init__(self, text_col_name: str,
                 min_df: int = 2,
                 max_df: float = 0.3):
        self.min_df = min_df
        self.max_df = max_df
        self.text_col_name = text_col_name

    def fit(self, df: DataFrame, y=None):
        pipe: Pipeline = Pipeline(
            steps=[('CountVectorizer',
                    CountVectorizer(stop_words='english',
                                    min_df=self.min_df,
                                    max_df=self.max_df,
                                    ngram_range=(1, 2))),
                   ('tfidftransformer', TfidfTransformer(use_idf=True))], verbose=False)
        pipe.fit(df[self.text_col_name])
        self.pipe = pipe
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        count_vect: CountVectorizer = self.pipe.steps[0][1]
        df_bow = pd.DataFrame(self.pipe.transform(df[self.text_col_name]).toarray(),
                              index=df.index,
                              columns=count_vect.get_feature_names())
        df_bow.columns = ['bow_' + col for col in df_bow.columns]  # rename in case of ints or protected terms
        df = pd.concat([df, df_bow], axis='columns')
        return df


class PreModelFilter(TransformerMixin):
    """Reduces a Dataframe on only the neede filters and drops all rows with missings"""

    column_names: Sequence[str]
    prefixes: Sequence[str]

    def __init__(self, column_names: Sequence[str], prefixes: Sequence[str]):
        self.column_names = column_names
        self.prefixes = prefixes

    def fit(self, df: DataFrame, y=None):
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        df_pred = df[[*[col for col in df.columns if col.startswith(tuple(self.prefixes))],
                      *self.column_names]].dropna()
        return df_pred
