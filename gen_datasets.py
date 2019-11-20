import pandas as pd
import numpy as np

import sys

if __name__ == '__main__':
    df = pd.read_excel('./data/agency_samples.xlsx')
    df = df[['sen_text', 'agency_cat']]
    np.random.shuffle(df.values)
    train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    train.to_excel('./data/train_df.xlsx')
    test.to_excel('./data/test_df.xlsx')
    validate.to_excel('./data/dev_df.xlsx')

