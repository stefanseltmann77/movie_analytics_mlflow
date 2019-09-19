import os
import mlflow
from mlflow.projects import LocalSubmittedRun
from sklearn.model_selection import ParameterGrid
from  mlflow.tracking import MlflowClient
client = MlflowClient()

grid = ParameterGrid({'term_confidence': range(1,2, 2), 'n_estimators': range(1, 20, 10)})

os.environ["MLFLOW_CONDA_HOME"] = "/home/ubuntu/anaconda3"
os.environ["MLFLOW_TRACKING_URI"] = "/home/ubuntu/envs/movies_analytics/bin/mlruns"

id = mlflow.create_experiment("my_exp6", "/home/ubuntu/envs/movies_analytics/bin/mlruns")

for i in grid:
    ml_run: LocalSubmittedRun  = mlflow.projects.run("/home/ubuntu/projects/movie_analytics", use_conda=True, backend="local",
                        experiment_id=id,
                        parameters={'target': 'is_genre_horror',
                                    'term_confidence': i['term_confidence'],
                                    'n_estimators': i['n_estimators'],
                                    'use_tfidf': 'yes'})

    client.set_tag(ml_run.run_id, "my_tag", "valid")
