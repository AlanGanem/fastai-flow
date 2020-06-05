from pathlib import Path
import pickle
import os
import time
import pandas as pd
from fastai.tabular import load_learner

def make_folder(folder_path, folder_name):

    folder_path = Path(folder_path)/Path(folder_name)
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print('{} already exists'.format(folder_path))
    return

def save_object(obj, saving_path, file_name):
    file_name = f'{file_name}'
    file_path = Path(saving_path)/file_name
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
    return

def make_fastai_serializable(learner):
    path = learner.path
    now = int(time.time())
    temp_file_name = f'temp_learner_{str(now)}.pkl'
    learner.export(temp_file_name, destroy = False)
    serializable_learner = load_learner(path,temp_file_name)
    os.remove(os.path.join(path, temp_file_name))
    return serializable_learner

def export_dict_as_folder(dict, saving_path, folder_name):
    saving_path = Path(saving_path)
    make_folder(saving_path, folder_name)
    for key in dict:
        if dict[key].__class__ in [pd.Series,pd.DataFrame]:
            dict[key].to_csv(saving_path/folder_name/f'{key}.csv')
        else:
            pass