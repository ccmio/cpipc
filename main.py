from prediction_model import *
import numpy as np
from ga import *
import pandas as pd
DATA_PATH = 'data/data.xlsx'
DATA_S_PATH = 'data/data_s.xlsx'
DATA_TOTAL_PATH = 'data/data_total.xlsx'
MODEL_PATH = './model/forest_reg.pkl'
MODEL_S_PATH = './model/forest_reg_S.pkl'
INPUT_PATH = ''

RE_TRAIN = False
RE_TRAIN_S = False


def main():
    pre_model = PredModel(DATA_PATH, MODEL_PATH)
    if RE_TRAIN:
        pre_model.train()
    pre_model_s = PredModel(DATA_S_PATH, MODEL_S_PATH)
    if RE_TRAIN_S:
        pre_model_s.train()

    # pre_model.predFunc(pd.read_excel(DATA_PATH), 1)
    # pre_model_s.predFunc(pd.read_excel(DATA_S_PATH), 0)
    opt_model = GA(DATA_TOTAL_PATH)
    opt_model.run()


if __name__ == '__main__':
    main()
