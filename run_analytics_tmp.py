import logging
from pathlib import Path

import pandas as pd

from movie_analytics.dataloader import DataLoader
# from movie_analytics.models.predict_top250_decisiontree import PredictTop250

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 999)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from movie_analytics.conf import project_dir
    print(project_dir)

    # dataloader = DataLoader()
    # df = dataloader.load_prepared_data(Path(r'./data/movie_prof.csv'))
    # print(df.shape)
    #
    # top250 = PredictTop250(dataloader)
    # top250.predict(df)
