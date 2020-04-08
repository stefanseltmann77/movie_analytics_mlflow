import logging
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from wordcloud import WordCloud

from movie_analytics.conf import project_dir


class OscarTextAnalytics:
    logger: logging.Logger
    chart_dir: Path

    def __init__(self, chart_dir=Path(project_dir, 'charts')):
        self.logger = logging.getLogger(__name__)
        self.chart_dir = chart_dir

    def create_wordcloud(self, df_words: pd.DataFrame, label: str):
        self.logger.info(f"Plotting wordcloud for {label}")
        wordcloud = WordCloud(width=480, height=480, margin=0). \
            fit_words(frequencies=df_words.sum().sort_values().to_dict())
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        chart_path = self.chart_dir / Path(f"word_cloud_{label}.png")
        plt.savefig(chart_path)
        return chart_path

    def create_importance_plot(self,
                               df_importance: DataFrame,
                               df_words_expl: DataFrame,
                               display_limit: int = 20):
        self.logger.info(f"Plotting feature importance for top {display_limit} features")
        df_lift: DataFrame = pd.concat([df_importance, df_words_expl], axis=1, sort=False). \
                                 sort_values('term', ascending=False)[:display_limit]
        df_lift['lift'] = df_lift[1] / df_lift[0]
        df_lift = df_lift[['term', 'lift']]

        df_lift.plot(kind='scatter', x='term', y='lift', color='royalblue')
        for i in range(df_lift.shape[0]):
            row = df_lift.iloc[i]
            plt.annotate(row.name, (row.term + 0.0005, row.lift))
        plt.ylabel("LIFT, measured in relative item frequency")
        plt.xlabel("feature importance")
        plt.title(f"The {display_limit} most important features")
        plt.xlim(df_lift.term.min() * 0.9, df_lift.term.max() * 1.1)
        plt.tight_layout()
        chart_path = self.chart_dir / Path(f'feature_importance.png')
        plt.savefig(chart_path)
        return chart_path

    def build_bag_of_words(self, df_movies: DataFrame, term_confidence: int, use_tfidf: bool) -> DataFrame:
        """build word matrix with counts"""
        vectorizer = CountVectorizer(stop_words='english', min_df=term_confidence, max_df=0.3, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(df_movies.storyline)
        feature_names = vectorizer.get_feature_names()
        if use_tfidf:
            # build word matrix with tfidf_counts
            vectorizer_tfidf = TfidfTransformer(use_idf=True)
            matrix = vectorizer_tfidf.fit_transform(matrix)
        # datasets, plain an tfidf
        df_words = pd.DataFrame(matrix.toarray(), columns=feature_names, index=df_movies.index)
        return df_words
