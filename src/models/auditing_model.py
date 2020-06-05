import sys
sys.path.append("..")
import pandas as pd
import difflib
from fastai.tabular import *
import os
import json
from features.build_features import build_features

import model_utils


def auditing_model(

        PRED_DATA_PATH = '../../data/external/history.csv',
        FITTED_OBJ_FOLDER = '../../models/',
        FITTED_OBJ_FILE = 'iva_model.pkl',
        OUTPUT_FILE_PATH = '../../models/auditing_outputs',
        pd_sep = ';',
        pd_encoding = 'ansi'
):
    """
    predict IVAs and append results and probas to source file
    :param PRED_DATA_PATH:
    :param FITTED_OBJ_FOLDER:
    :param FITTED_OBJ_FILE:
    :param OUTPUT_FILE_PATH:
    :return:
    """
    NOW = str(pd.Timestamp.now().timestamp()).replace('.','_')
    PRED_DATA_PATH = os.path.abspath(PRED_DATA_PATH)
    FITTED_OBJ_FOLDER = os.path.abspath(FITTED_OBJ_FOLDER)
    OUTPUT_FILE_PATH = os.path.abspath(OUTPUT_FILE_PATH + r'\{}'.format(NOW))

    #create predictions directory
    os.mkdir(OUTPUT_FILE_PATH)

    #load fitted_model
    inf_learner = load_learner(path = FITTED_OBJ_FOLDER, file = FITTED_OBJ_FILE)
    df = pd.read_csv(PRED_DATA_PATH, encoding = pd_encoding, sep = pd_sep)

    # build_features
    df = build_features(df)

    #audit
    top_losses_df = model_utils.top_losses(model = inf_learner, df = df)
    top_losses_df.reset_index()

    # save to excell
    top_losses_df.to_excel(r'{}/{}'.format(OUTPUT_FILE_PATH, 'auditting.xlsx'))
    #IMPLEMENT PREDICTION METADATA USING FITTEDMODEL METADATA
    #with open(r'{}/{}/{}'.format(OUTPUT_FILE_PATH,NOW,'model_metadata.txt'), 'w') as file:
    #    file.write(json.dumps(fitted_model_meta))

    return

if __name__ == '__main__':
    auditing_model()