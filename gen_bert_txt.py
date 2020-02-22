import pandas as pd
import sys

if __name__ == '__main__':
    dataset, method, pvalue = sys.argv[1], sys.argv[2], sys.argv[3]
    path = './gen_sen/res_sen_'+ dataset + '_' + method+'.csv' 
    df = pd.read_csv(path)
    pvalue = float(pvalue)
    df = df[df['p-value'] == pvalue]
    if dataset == 'para':
        sen = 'sen0'
    else:
        sen = 'sen'
    res_path = './gen_sen/txt_' + dataset + '_' + method
    ori_path = res_path + '_ori.txt'
    df[sen].to_csv(ori_path, header=False, index=False)
    df['out'].to_csv(res_path + '_out.txt', header=False, index=False)
