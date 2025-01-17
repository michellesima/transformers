import pandas as pd
import sys

def max_num_bigram(sen):
    toks = sen.split()
    if len(toks) < 2:
        return 0
    dct = {}
    for i in range(len(toks) - 1):
        dct[tuple(toks[i:i+1])] = dct.get(tuple(toks[i:i+1]), 0) + 1
    return max(dct.values())

if __name__ == '__main__':
    ds, method, thres = sys.argv[1], sys.argv[2], int(sys.argv[3])
    df = pd.read_csv('gen_sen/res_sen_' + ds + '_' + method + '.csv')
    print(len(df.index))
    groups = df.groupby(by='p-value')
    for p, subdf in groups:
        lst = subdf['out'].tolist()
        len_lst = [max_num_bigram(sen) for sen in lst]
        ori_len = len(len_lst)
        len_lst = list(filter(lambda x: x > thres, len_lst))
        pct = len(len_lst) / ori_len
        print('with p value ', p, ' percent greater than ', str(thres), 'bigrams in a sentence is ', str(pct))
