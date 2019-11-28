# -*- coding: utf-8 -*-
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd
from pandas import np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from oscar_analytics import OscarAnalytics
from conf.configs import configs

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 999)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # OSCARS_MIN = 3
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in sys.argv[1:]}

    target_feature = args.get('target', 'is_oscar_winner')
    term_confidence = int(float(args.get('term_confidence', 3)))
    use_tfidf = args.get('use_tfidf', 'yes') == 'yes'
    n_estimators = int(args.get('n_estimators', 10))

    mlflow.set_tracking_uri(configs.get("paths", "mlruns"))
    mlflow.set_experiment("MovieAnalytics")

    with mlflow.start_run():
        mlflow.log_params({'use_tfidf': use_tfidf, 'target_feature': target_feature})
        oa = OscarAnalytics(Path("./dataset/movie_prof.csv"))
        df_movies = oa.load_data()
        df_movies = oa.clean_data(df_movies)
        df_movies = oa.add_ranking_flags(df_movies)

        # build word matrix with counts
        vectorizer = CountVectorizer(stop_words='english', min_df=term_confidence, max_df=0.3, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(df_movies.storyline)
        feature_names = vectorizer.get_feature_names()

        if use_tfidf:
            # build word matrix with tfidf_counts
            vectorizer_tfidf = TfidfTransformer(use_idf=True)
            matrix = vectorizer_tfidf.fit_transform(matrix)

        # datasets, plain an tfidf
        df_words = pd.DataFrame(matrix.toarray(), columns=feature_names, index=df_movies.index)
        df_words.sum().sort_values().to_dict()

        # target = df_movies.award_noms_oscar >= OSCARS_MIN
        target = df_movies[target_feature]

        oa.create_wordcloud(df_words.loc[target == 0], 'nontarget')
        oa.create_wordcloud(df_words.loc[target == 1], 'target')
        classifier = \
            RandomForestClassifier(n_estimators=n_estimators,
                                   min_samples_split=50,
                                   min_samples_leaf=15,
                                   max_depth=3). \
                fit(df_words, target)

        cv = np.median(cross_val_score(classifier, df_words, target, scoring='roc_auc', cv=10))
        auc = roc_auc_score(target, classifier.predict_proba(df_words)[:, 1])
        mlflow.log_metric('auc_training', auc)
        mlflow.log_metric("auc_cv10_median", cv)

        df_importance = pd.DataFrame(classifier.feature_importances_, columns=['term'], index=feature_names)
        df_words_expl = df_words# .copy()
        df_words_expl.loc[:, 'target'] = target
        df_words_expl = df_words_expl.groupby("target").mean().transpose()

        oa.create_importance_plot(df_importance, df_words_expl)

        mlflow.log_artifacts('./charts')
        del df_words_expl
        del df_movies

        # mlflow run https://github.com/stefanseltmann77/movie_analytics_mlflow.git --experiment-name=MovieAnalytics