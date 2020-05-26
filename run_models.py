import os

import mlflow
from mlflow.projects import LocalSubmittedRun
from sklearn.model_selection import ParameterGrid

from movie_analytics.conf.configs import configs

grid = ParameterGrid({'term-confidence': (0.0001, 0.001, 0.01),
                      'max-depth': (5, 8, 10, 20),
                      'n-estimators': (5, 10, 50, 100, 200)})

os.environ["MLFLOW_CONDA_HOME"] = "/home/sseltmann/miniconda3"
os.environ["MLFLOW_TRACKING_URI"] = configs.get("paths", "mlruns")


if __name__ == "__main__":
    EXPERIMENT_NAME: str = 'Oscar_Analytics_BOW'

    for i in grid:
        # ml_run: LocalSubmittedRun = mlflow.projects.run(str(Path(project_dir)),
        ml_run: LocalSubmittedRun = mlflow.projects.run('/home/sseltmann/movie_analytics',
                                                        backend="local",
                                                        use_conda=True,
                                                        entry_point='run_bow_classifier.py',
                                                        experiment_name=EXPERIMENT_NAME,
                                                        parameters={'target-feature': 'is_genre_scifi',
                                                                    'use-tfidf': 'yes', **i})

    # client.set_tag(ml_run.run_id, "my_tag", "valid")

    # client = MlflowClient()
    # id = mlflow.create_experiment("oscar_analytics", os.environ["MLFLOW_TRACKING_URI"])
    # id = mlflow.get_experiment_by_name("oscar_analytics")
    # id = mlflow.get_experiment_by_name("oscar_analytics").experiment_id

    # id = mlflow.create_experiment("oscar_analytics", "/home/ubuntu/envs/movies_analytics/bin/mlruns")
