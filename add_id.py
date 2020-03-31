import pandas as pd
import glob, os

df_ori = pd.read_excel('./gen_sen/agency_samples.xlsx')
dropcol = ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 0.1']
df_ori.drop(labels=dropcol, inplace=True, axis=1)
dropcol.remove('Unnamed: 1')
dropcol.remove('Unnamed: 0.1')
df_ori.drop_duplicates(inplace=True)
df_ori.rename(mapper={'sen_text': 'sen'}, inplace=True, axis=1)
os.chdir("./data/roc")
tem = df_ori.groupby('sen')
for file in glob.glob("*.csv"):
    df = pd.read_csv(file)
    df.drop(labels=dropcol, inplace=True, axis=1)
    print(df.head())
    colres = list(df.columns)
    colres.append('storyid')
    colres.append('sentencenum')
    print(len(df.index))
    new_df = df.merge(right=df_ori, how='left', on='sen')
    print(new_df.columns)
    print(colres)
    new_df = new_df[colres]
    new_df.to_csv(file)
    print(len(new_df.index))
    print(file)
