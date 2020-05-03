import configparser
from pathlib import Path

from ..conf import project_dir

configs = configparser.ConfigParser()

configs.read_dict({'paths': {'mlruns': f'http://3.122.204.165:5000/',
                             'dataset': Path(project_dir, "data/movie_prof.csv")}})
