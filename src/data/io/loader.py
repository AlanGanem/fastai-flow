import pandas as pd
import pickle
from pathlib import Path

def load_csv(path, encoding = 'utf-8', sep = ';'):
    path = Path(r'{}'.format(path))
    data = pd.read_csv(path, sep = sep, encoding = encoding)
    return data

def save_csv(data, folder_path, file_name):
    folder_path = Path(r'{}'.format(folder_path))
    file_path = folder_path / file_name
    data.to_csv(file_path)
    return file_path

def load_object(path):
    path = Path(path)
    with (open(path, "rb")) as openfile:
        obj = pickle.load(openfile)
    return obj

