import os
import mlflow
from mlflow.projects import LocalSubmittedRun
from sklearn.model_selection import ParameterGrid
from  mlflow.tracking import MlflowClient

grid = ParameterGrid({'term_confidence': range(1,2, 2), 'n_estimators': range(1, 20, 10)})

mlruns_folder = "/home/ubuntu/mlruns"

os.environ["MLFLOW_CONDA_HOME"] = "/home/ubuntu/anaconda3"
os.environ["MLFLOW_TRACKING_URI"] = mlruns_folder


if __name__ == "__main__":
    client = MlflowClient()
    # id = mlflow.create_experiment("oscar_analytics", os.environ["MLFLOW_TRACKING_URI"])
    # id = mlflow.get_experiment_by_name("oscar_analytics")
    # id = mlflow.get_experiment_by_name("oscar_analytics").experiment_id

    # id = mlflow.create_experiment("oscar_analytics", "/home/ubuntu/envs/movies_analytics/bin/mlruns")

    for i in grid:
        ml_run: LocalSubmittedRun  = mlflow.projects.run("/home/ubuntu/projects/movie_analytics", use_conda=True, backend="local",
                            experiment_name="oscar_analytics",
                            parameters={'target': 'is_genre_horror',
                                        'term_confidence': i['term_confidence'],
                                        'n_estimators': i['n_estimators'],
                                        'use_tfidf': 'yes'})

        client.set_tag(ml_run.run_id, "my_tag", "valid")
