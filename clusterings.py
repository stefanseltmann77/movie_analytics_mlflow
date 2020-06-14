from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from movie_analytics.conf.configs import configs
from movie_analytics.dataloader import DataLoader
from movie_analytics.transformers import BagOfWords, PreModelFilter

if __name__ == '__main__':
    dataloader = DataLoader()

    boxes = pd.read_csv("M:/gesichert/Programming/python/media/exports/storage.tsv", index_col='imdb_movie_id')
    boxes.rename(columns={'storage': 'box'}, inplace=True)

    df_movies = dataloader.load_prepared_data(configs.get("paths", "dataset"))

    pipe = make_pipeline(BagOfWords(text_col_name='storyline'),
                         PreModelFilter(column_names=['rating_imdb', 'year', 'runtime', 'budget'],
                                        prefixes=('bow_', 'is_genre_')),
                         RandomForestClassifier(n_estimators=200))

    df_movies_wthbox = df_movies.join(boxes)
    df_movies_wthbox.dropna(inplace=True)

    pipe.fit(X=df_movies_wthbox, y=df_movies_wthbox.box)

    # pipe.predict_proba(df_movies)

    # pipe.steps[-1][1].classes_
    dump(pipe, Path('C:/Users/sselt/PycharmProjects/movie_analytics_mlflow/data') / Path('storage.pipe'))

    # boxes_ids = list(df_movies_wthbox.loc[df_movies_wthbox.box != 'undefined'].box.index)
    #
    # rfc.fit(df_train.loc[df_train.index.intersection(boxes_ids)],
    #         y=df_movies_wthbox.loc[df_train.index.intersection(boxes_ids)].box)
    #
    #
    # df_predict =  pipe.transform(df_test)
    #
    # rfc.predict_proba(df_predict)
    #
    #
    # df_movies_wthbox = bow.fit_transform(X=df_movies_wthbox)
    #
    # df_pred = build_dataset.transform(df_movies_wthbox)
    #
    #
    # rfc = RandomForestClassifier(n_estimators=200)
    # rfc.fit(df_pred.loc[df_pred.index.intersection(boxes_ids)],
    #         y=df_movies_wthbox.loc[df_pred.index.intersection(boxes_ids)].box)
    #
    # rfc.classes_
    #
    # df_result = pd.DataFrame(rfc.predict_proba(df_pred.loc[df_pred.index.intersection(boxes_ids)]),
    #                          columns=rfc.classes_)
    #
    # build_dataset.transform(bow.transform(df_test))
    # rfc.predict_proba(build_dataset.transform(build_dataset.transform(bow.transform(df_test))))
    #
    # df_result = pd.DataFrame(rfc.predict_proba(df_pred), columns=rfc.classes_).set_index(df_pred.index)
    # df_movies_wthbox.index.value_counts()
    #
    # df_result['title'] = df_movies_wthbox.title
    # df_result['box'] = df_movies_wthbox.box
    #
    # pd.concat([df_result, df_movies_wthbox.loc[df_pred.index.intersection(boxes_ids)].
    #           reset_index()[['index', 'title', 'box']]],
    #           axis='columns'). /
    #     dropna().sort_values('horror')
    #
    # df_result.sort_values('western')
    #
    # df_movies_wthbox.box.value_counts()
    #
    # df_result = pd.DataFrame(rfc.predict_proba(df_pred), columns=rfc.classes_)
    # rfc.classes_
    # pd.set_option('min_rows', 80)
    # pd.set_option('display.width', 300)
    # pd.set_option('display.max_columns', 10)
    #
    # pd.concat([df_result, df_movies.loc[df_pred.index].reset_index()[['imdb_movie_id', 'title']]], axis=1).sort_values(
    #     "fantasy")[['imdb_movie_id', 'title']]
    #
    # df_pred
    # df_result
    # df_movies.loc[df_pred.index].reset_index()[['imdb_movie_id', 'title']]
    #
    # df_movies
    # df_pred[['index', 'title']]
    #
    # df_pred.index.intersection()
    #
    # df_pred
    #
    # df_tmp.sort_values(by='title').to_dict(orient='records')
    #
    # mlflow.set_tracking_uri(configs.get("paths", "mlruns"))
    # # mlflow.set_experiment("MovieAnalytics_Clusters")
    #
    # # mlflow_pyfunc_model_path = "/home/sseltmann/models/clustering8"
    #
    # mod = MovieNeighbors(df_movies)
    # from mlflow.sklearn import log_model
    #
    # log_model(mod, 'cluster_log')
    #
    # cvectorizer = CountVectorizer(stop_words='english',
    #                               min_df=0.1,
    #                               max_df=0.3,
    #                               ngram_range=(1, 2))
    #
    # cvectorizer = CountVectorizer(stop_words='english', binary=False, min_df=1, max_df=0.1)
    # bagofwords = cvectorizer.fit_transform(df_movies.storyline).toarray()
    #
    # df_bow = pd.DataFrame(bagofwords, columns=cvectorizer.get_feature_names())
    # # df_tmp = df_bow[['war', 'zombies', 'life']].loc[(df_bow.zombie > 0)  | (df_bow.war > 0)]
    #
    # # df_bow_tdif = TfidfTransformer(use_idf=True).fit_transform(df_bow).toarray()
    # # df_tmp
    # # df_bow_tdif
    #
    # dist_matr_storyline = chi2_kernel(df_bow)
    # pd.DataFrame(dist_matr_storyline).stack().sum()
    # # 1384
    # # 1445
    # # write optimizer for height of sum.
    # list(df_movies.title)
    # df_tmp = pd.DataFrame(dist_matr_storyline, index=list(df_movies.title), columns=list(df_movies.title))
    # df_tmp['The Matrix'].sort_values()
    #
    # cvectorizer.get_feature_names()
    #
    # df_stacked = pd.DataFrame(dist_matr_storyline).stack()
    # df_stacked
    #
    # df_stacked.loc[df_stacked < 1].plot(kind='hist', bins=500)
    # import matplotlib.pyplot as plt
    #
    # plt.show()
    #
    # ies.storyline.str.len()
    # pipe.fit(df_movies.storyline)
    #
    # pipe.transform(df_movies.storyline).toarray()
    #
    # co
    #
    # dist_matr_storyline = chi2_kernel(pipe.transform(df_movies.storyline).toarray())
    #
    # model = load_model('cluster_log')
    #
    # #
    # # save_model(path=mlflow_pyfunc_model_path, python_model=MovieNeighbors(df_movies),
    # #            conda_env='/home/sseltmann/movie_analytics/movie_analytics/conf/conda_env.yml',
    # #            artifacts=artifacts)
    # #
    # # mod
    # # mod = load_model(mlflow_pyfunc_model_path)
    # # print(mod)
    # # df_imput =  pd.DataFrame([{"imdb_movie_id": 133093}]).reset_index()
    # # print(mod.predict(df_imput))
    #
    # # print(mn.predict(None, df_imput))
    #
    # #
    #
    # # distdf_genre = mn.build_distance_matrix(df)
    # # print(distdf_genre.nsmallest(10, 15864))
    #
    # # import mlflow
    # # mlflow.pyfunc.log_model('clustermod', python_model=mn)
    # #
    # # mlflow.pyfunc.load_model('clustermod')
    # #
    # #
    #
    # """
    # curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json; format=pandas-records' -d '[{"imdb_movie_id": 133093}]'
    #
    #
    #
    # """
    #
    # from sklearn.cluster import AgglomerativeClustering
    #
    # dist_matrix = mod.build_distance_matrix(mod.df)
    # dist_matrix
    #
    # dist_matr_storyline = pd.DataFrame(dist_matr_storyline, index=df_movies.title)
    # dist_matr_storyline
    #
    # clustering = AgglomerativeClustering(n_clusters=99).fit(dist_matr_storyline)
    #
    # # print the class labels
    # print(clustering.labels_)
    #
    # cluster_labels = pd.Series(clustering.labels_)
    # dist_matr_storyline.reset_index().loc[cluster_labels == 1]
    #
    # cluster_labels == 1
    #
    # dist_matrix.iloc[cluster_labels == 1]
    #
    # clustering.labels_
    #
    # print("abc")
