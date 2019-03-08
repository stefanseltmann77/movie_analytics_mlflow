# -*- coding: utf-8 -*-
import mlflow.pyfunc


mlflow.pyfunc.save_model('/home/btraining/PycharmProjects/oscarmodel_trees8',
                         conda_env='conda_env.yml',
                         loader_module='simple_model',
                         data_path='./dataset/movie_prof.csv',
                         code_path=['/home/btraining/PycharmProjects/oscar_analytics']
                         )