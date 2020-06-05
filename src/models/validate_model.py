import pandas as pd
from fastai import *
from fastai.tabular import *
from predict_model import get_preds
import difflib
import model_utils

def validate_model(model, data,VALIDATION_PATH):

    validation_dict = model_utils.validation_dict(model,data)
    for item in validation_dict:
        if 'df' in item:
            # create excel spreadsheet
            validation_dict[item].to_excel(r'{}/{}_{}.xlsx'.format(VALIDATION_PATH,item))

