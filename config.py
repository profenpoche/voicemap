import os

PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = '/home/profenpoche/voicemap/data'

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
