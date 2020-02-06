import pandas as pd

if __name__ == '__main__':
    df = pd.read_excel('data/dev_df.xlsx')
    print(df.head())
