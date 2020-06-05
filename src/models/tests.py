from model_utils import validation_dict
import pandas as pd
from fastai import *
from fastai.tabular import *
import sys
sys.path.append("..")
from features.build_features import build_features
from train_model import train_model
from predict_model import predict_model

def test_main():
    train_model(val_days = 30,test_days = 21, validate = False, fastai_cycles = 5)
    predict_model(PRED_DATA_PATH = '../../data/external/history.csv')


if __name__ == '__main__':
    test_main()