import logging

import click
import mlflow.sklearn
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from movie_analytics.conf.configs import configs
from movie_analytics.dataloader import DataLoader
from oscar_analytics import OscarTextAnalytics

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 999)


@click.command(name='run')
@click.option("--target-feature", type=click.STRING, default="is_oscar_winner")
@click.option("--term-confidence", type=click.FLOAT, default=1)
@click.option("--use_tfidf", type=click.BOOL, default=True)
@click.option("--n-estimators", type=click.INT, default=100)
def run_bow_classifier(target_feature, term_confidence, use_tfidf, n_estimators):
    mlflow.set_tracking_uri(configs.get("paths", "mlruns"))

    mlflow.set_experiment("MovieAnalytics_Texts")

    with mlflow.start_run():
        mlflow.log_params({'use_tfidf': use_tfidf,
                           'target_feature': target_feature,
                           'term_confidence': term_confidence,
                           'n_estimators': n_estimators})

        dataloader = DataLoader()
        oa = OscarTextAnalytics()

        df_movies = dataloader.load_prepared_data(configs.get("paths", "dataset"))
        target: Series = df_movies[target_feature]

        pipe: Pipeline = Pipeline(
            steps=[('CountVectorizer',
                    CountVectorizer(stop_words='english',
                                    min_df=term_confidence,
                                    max_df=0.3,
                                    ngram_range=(1, 2))),
                   ('classifier', RandomForestClassifier(n_estimators=n_estimators,
                                                         min_samples_split=20,
                                                         min_samples_leaf=15,
                                                         max_depth=3))], verbose=False)
        if use_tfidf:
            pipe.steps.insert(len(pipe) - 1, ('tfidftransformer', TfidfTransformer(use_idf=True)))

        pipe.fit(df_movies.storyline, y=target)
        feature_importances = pipe.steps[-1][1].feature_importances_

        cv = np.median(cross_val_score(pipe, df_movies.storyline, target, scoring='roc_auc', cv=10))
        auc = roc_auc_score(target, pipe.predict_proba(df_movies.storyline)[:, 1])

        # get the transformed data:
        pipe = pipe.set_params(classifier='passthrough')
        bagofwords_raw = pipe.transform(df_movies.storyline)
        feature_names = pipe.named_steps['CountVectorizer'].get_feature_names()
        df_bagofwords: DataFrame = pd.DataFrame(bagofwords_raw.toarray(), columns=feature_names, index=df_movies.index)

        oa.create_wordcloud(df_bagofwords.loc[target == 1], 'target')
        oa.create_wordcloud(df_bagofwords.loc[target != 1], 'nontarget')

        df_importance = pd.DataFrame(feature_importances, columns=['term'], index=df_bagofwords.columns)
        df_words_expl = df_bagofwords
        df_words_expl.loc[:, 'target'] = target
        df_words_expl = df_words_expl.groupby("target").mean().transpose()
        oa.create_importance_plot(df_importance, df_words_expl)

        mlflow.log_metrics({'auc_training': auc,
                            "auc_cv10_median": cv})
        mlflow.log_artifacts(oa.chart_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_bow_classifier()
