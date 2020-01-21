import pandas as pd
import sys
import string

def strip_puc(sen):
    i = 0
    while i < len(sen) and sen[i] in string.punctuation:
        i += 1
    return sen[i: ]

if __name__ == '__main__':
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
