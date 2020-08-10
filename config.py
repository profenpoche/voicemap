import os

PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = '/home/profenpoche/voicemap/data'
LOG_PATH = '/home/profenpoche/voicemap/logs'

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
