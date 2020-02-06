import pandas as pd
import sys
import string
from utils import *

def strip_puc(sen):
    i = 0
    while i < len(sen) and sen[i] in string.punctuation:
        i += 1
    return sen[i: ]

def split_para():
    randind = 7
    df = pd.read_csv('./data/parads/senp_bs.zip')
    df['sen0'].apply(strip_puc)
    df['sen1'].apply(strip_puc)
    print(len(df.index))
    train = df.sample(n=45000, random_state=randind)
    df = df.drop(train.index)
    test = df.sample(n=10000, random_state=randind)
    dev = df.drop(test.index)
    print(len(dev.index))
    train.to_csv('./data/parads/senp_bs_train.zip', compression='zip')
    test.to_csv('./data/parads/senp_bs_test.zip', compression='zip')
    dev.to_csv('./data/parads/senp_bs_dev.zip', compression='zip')

def split_roc():
    randind = 7
    df = pd.read_excel('./data/agency_samples.xlsx')
    df.rename(mapper={'sen_text': 'sen'}, axis='columns', inplace=True)
    df['sen'].apply(strip_puc)
    print(len(df.index))
    train = df.sample(n=12000, random_state=randind)
    df = df.drop(train.index)
    dev = df.sample(n=2000, random_state=randind)
    test = df.drop(dev.index)
    print(len(test.index))
    train.to_csv(ROC_TRAIN)
    dev.to_csv(ROC_DEV)
    test.to_csv(ROC_TEST)

if __name__ == '__main__':
    split_roc()
