import logging
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from pandas import np
from wordcloud import WordCloud

from listings import oscar_winners, top_250_engl, worst_100


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

    def create_wordcloud(self, df_words: pd.DataFrame, label: str):
        self.logger.info(f"Plotting Wordcloud for {label}")
        wordcloud = WordCloud(width=480, height=480, margin=0). \
            fit_words(frequencies=df_words.sum().sort_values().to_dict())
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.savefig(f"./charts/word_cloud_{label}.png")

    def create_importance_plot(self, df_importance, df_words_expl, display_limit: int=20):
        self.logger.info(f"Plotting feature importance for top {display_limit} features")
        df_lift = pd.concat([df_importance, df_words_expl], axis=1, sort=False).\
                      sort_values('term', ascending=False)[:display_limit]
        df_lift['lift'] = df_lift[1] / df_lift[0]
        df_lift = df_lift[['term', 'lift']]

        df_lift.plot(kind='scatter', x='term', y='lift', color='royalblue')
        for i in range(df_lift.shape[0]):
            row = df_lift.iloc[i]
            plt.annotate(row.name, (row.term+0.0005, row.lift))
        plt.ylabel("LIFT, measured in relative item frequency")
        plt.xlabel("feature importance")
        plt.title(f"The {display_limit} most important features")
        plt.xlim(df_lift.term.min()*0.9, df_lift.term.max()*1.1)
        plt.tight_layout()
        plt.savefig(f"./charts/feature_importance.png")
