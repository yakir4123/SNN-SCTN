from os import listdir
from os.path import isfile, join as pjoin


def get_raw_files(path: str):
    return [pjoin(path, f) for f in listdir(path) if f.endswith('raw')]




