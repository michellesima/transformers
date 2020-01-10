import pandas as pd

if __name__ == '__main__':
    randind = 7
    df = pd.read_csv('./data/parads/senp_bs.zip')
    train = df.sample(n=40000, random_state=randind)
    df = df.drop(train.index)
    test = df.sample(n=10000, random_state=randind)
    dev = df.drop(test.index)
    print(len(dev.index))
    train.to_csv('./data/parads/senp_bs_train.zip', compression='zip')
    test.to_csv('./data/parads/senp_bs_test.zip', compression='zip')
    dev.to_csv('./data/parads/senp_bs_dev.zip', compression='zip')
