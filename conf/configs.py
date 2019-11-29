import configparser

configs = configparser.ConfigParser()
configs.read_dict({"paths": {"mlruns": "./mlruns"}})
