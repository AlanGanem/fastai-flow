import sys
sys.path.append("..")
import pandas as pd
import json
import os

from fastai.tabular import *
from fastai import *
# internal modules
from .validation import classification_validation
from .data.io.loader import load_csv
from .preprocessing.consistency import classification_consistency_train_val
from .preprocessing.preprocess import drop_dependent_nan, df_split
from .models.model_utils import create_multiclass_db, create_multiclass_learner, fit_learner

#from validate_model import validate_model
import model_utils



def train_model(
        train_data,
        cat_features,
        num_features,
        dependent_vars = ['IVAMIRO'],
        fastai_layers_setup = [20],
        fastai_dropout = 0.1,
        cat_emb_szs = None,
        fastai_cycles = 1,
        fastai_bs = 256,
        fastai_patience = 3,
        fastai_min_delta = 0.001,
        validate = True,
        # loading args
        pd_sep = ';',
        pd_encoding = 'ansi',
        #  split args                        
        val_days = 21,
        test_days = 14,        
):
    
    NOW = pd.to_datetime('today')
    NOWTS = str(NOW.timestamp).split('.')[0]
    NOWDT = str(pd.to_datetime('today').date())
    #set paths:    
    SAVING_PATH = '../../models'    
    MODEL_NAME = '{}_model.pkl'.format(NOWTS)    
    
    MODEL_FULL_PATH = SAVING_PATH + MODEL_NAME
    #get abboslute path to save serialized model and metadata
    SAVING_PATH = os.path.abspath(SAVING_PATH)
    MODEL_FULL_PATH = os.path.abspath(MODEL_FULL_PATH)


    # load training data
    data = load_csv(TRAIN_DATA_PATH, encoding = pd_encoding, sep = pd_sep)
    
    ## preprocess
    #drop label nans
    data = drop_dependent_nan(data, dependent_vars)

    # build_features # \src\features\build_features - FEATURES SHOULD BE BUILT OUTSIDE PACKAGE
    #data = build_features(data)
    #split data
    train_data, test_data = df_split(data, train_frac=0.9, date_col = date_col, test_days=None, start_from=None, test_days=None, start_from=None)
    train_data,val_data = df_split(train_data, train_frac=0.9, date_col = date_col, test_days=None, start_from=None, test_days=None, start_from=None)

    print('train size: {}%\nvalidation size: {}%\ntest size: {}%'.format(
        round(train_date_msk.sum()/len(data),2)*100,
        round(val_date_msk.sum()/len(data),2)*100,
        round(test_date_msk.sum()/len(data),2)*100,))


        #
    
    ## set features
    #cat_features = list(cat_emb_szs) #TODO create optional param for embedding size setup (allow optimization)

    ## remove unseen (in train) labels from validation set # src\consistency\classification_consistency_train_val
    train_data, val_data = classification_consistency_train_val(train_data, val_data, dependent_vars)

    #create databunch
    db = create_multiclass_db(train_data, val_data, cat_features, num_features, dependenet_vars)

    # create learner
    learner = create_multiclass_learner(
            db,
            fastai_layers_setup,
            cat_emb_szs,
            fastai_dropout,
            fastai_min_delta,
            fastai_patience,
    )
    
    
    #fit learner
    fit_learner(learner, fastai_cycles)
    #export model
    learner.export(MODEL_FULL_PATH)
    
    
    
    # validate model
    if validate == True:
        validation_dict = {
            'train':classification_validation.validation_dict(learner, train_data, dependent_vars[0]),
            'validation': classification_validation.validation_dict(learner, val_data, dependent_vars[0]),
            'test': classification_validation.validation_dict(learner, test_data, dependent_vars[0]),
        }
    else:
        validation_dict = {
            'train': {'accuracy':{}},
            'validation': {'accuracy':{}},
            'test': {'accuracy':{}},

        }
    metadata = {
        'categorical_vars': str(cat_features),
        'dependent_vars': str(dependent_vars),
        'train_accuracy': validation_dict['train']['accuracy'],
        'validation_accuracy': validation_dict['validation']['accuracy'],
        'test_accuracy':validation_dict['test']['accuracy']
    }
    
    #create folder for metadata
    
    METADATA_PATH = r'{}/train_metadata/fitted_{}'.format(SAVING_PATH,NOWDT)
    try:
        os.mkdir(METADATA_PATH)
    except FileExistsError:
        print('{} already exists'.format(METADATA_PATH))
    # save model validation
    for data_set in validation_dict:
        for item in validation_dict[data_set]:
            if 'df' in item:
                # create excel spreadsheet
                validation_dict[data_set][item].to_excel(r'{}/{}_{}.xlsx'.format(METADATA_PATH,item,data_set))

    # save model metadata
    with open(r'{}/metadata.txt'.format(METADATA_PATH), 'w') as file:
        file.write(json.dumps(metadata))  # use `json.loads` to do the reverse
    # export model

    

    return

if __name__ == '__main__':
    train_model(
        pd_sep=',',
        pd_encoding='iso-8859-1',
        cat_emb_szs={
            'desc1': 4,
            'Material': 8,
            'Fornecedor': 6,
            'Filial': 4,
            'PEP': 2,
            'UF': 6,
            'InterUF': 1,
            'IVAPC': 6,
            'OrderType': 2,
        },
        fastai_cycles=5,
        fastai_patience=2,
        fastai_min_delta=0.002,
        fastai_layers_setup=[10],
        fastai_dropout=0,
        dependent_vars=['IVAMIRO'],
        val_days=28,
        test_days=28,
        validate=True,
        fastai_bs=1024,
    )