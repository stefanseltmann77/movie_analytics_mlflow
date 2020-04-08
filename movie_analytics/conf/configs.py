import configparser
from pathlib import Path

from ..conf import project_dir

configs = configparser.ConfigParser()
configs.read_dict({'paths': {'mlruns': Path(project_dir.parent, 'mlruns'),
                             'dataset': Path(project_dir, "data/movie_prof.csv")}})
