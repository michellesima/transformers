import pandas as pd

numepoch = 1

if __name__ == '__main__':
    df = pd.read_excel('./gen_sen/epoch_tem' + str(numepoch) + '.xlsx')

    newdf.to_excel('./gen_sen/format_epoch1.xlsx')
    print(newdf.head(10))
