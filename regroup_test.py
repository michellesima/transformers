import pandas as pd

numepoch = 1

if __name__ == '__main__':
    df = pd.read_excel('./gen_sen/epoch_tem' + str(numepoch) + '.xlsx')
    outdf = df['out'].str.split(pat='<end>', n=1, expand=True)
    newdf = df['ori'].str.split(pat='<cls>', n=1, expand=True)
    newdf.columns = ['sen', 'cat']
    newdf['outbe'] = outdf[0]
    newdf['cat'] = newdf['cat'].str.replace(' <cls>', '')
    newdf['out_sen'] = df['out']
    newdf = newdf.sort_values(by='sen')
    newdf.to_excel('./gen_sen/format_epoch1.xlsx')
    print(newdf.head(10))
