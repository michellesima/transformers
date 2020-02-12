from transformers import *
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils import *
from utils_dr import *
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from examples.run_generation import *
import sys

max_sen_len = 20
random_seed = 7
numepoch = 10
ps = [0.4, 0.6, 0.8, 0.9]
agen_v = agen_verbs()
VER_MAG_RATE = 1.5

def repeatN(list, n):
    ori = list
    for _ in range(n):
        list = list.append(ori, ignore_index=True)
    return list

def gen_p(model, test_dataset):
    outlist = []
    outp = []
    for i in ps:
        for sen in test_dataset:
            senlen = len(sen)
            out = sample_sequence(
                model=model,
                context=sen,
                length=max_sen_len,
                top_p=i,
                device=device_dr
            )
            out = out[0, senlen:].tolist()
            text = tokenizer_dr.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)

            end_ind = text.find('<end>')
            if end_ind >= 0:
                text = text[0: end_ind]
            outlist.append(text)
            outp.append(i)
    return outlist, outp

def eval_model(mind, test_dataset, df, mtd='para'):
    '''
    get generated sentence for a particular model
    '''
    finaldf = pd.DataFrame()
    if mtd == 'para':
        savedir = './modelp/savedmodels'
    elif mtd == 'mix':
        savedir = './modelmix/savedmodels'
    else:
        savedir = './modelr/savedmodels'
    if 'sen0' in df.columns:
        colsen = 'sen0'
    else:
        colsen = 'sen'
    modelpath = savedir + str(mind)
    print(modelpath)
    model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
    model.to(device_dr)
    model.eval()
    df = repeatN(df, len(ps) - 1)
    outlist, outp = gen_p(model, test_dataset)
    df['out'] = outlist
    df['p-value'] = outp
    df.sort_values(by=[colsen, 'p-value'], inplace=True)
    df['modelind'] = mind
    finaldf = finaldf.append(df, ignore_index=True)
    return finaldf

def gen_roc(mindi, model='para'):
    '''
    generate sen for roc stories
    :param mind: epoch of model
    :return:
    
    test_df = pd.read_excel('./data/dev_df.xlsx')
    test_df = test_df.head(10)
    # list of encoded sen
    test_dataset, df = make_dataset(test_df, train_time=False)
    finaldf = eval_model(mind, test_dataset, df)
    savedfile = 'gen_sen/res_sen.xlsx'
    finaldf.to_excel(savedfile)
    '''
    test_dataset, df = parse_file_dr(ROC_DEV, train_time=False)
    print(df.columns)
    finaldf = eval_model(mind, test_dataset, df, mtd=model)
    savedfile = 'gen_sen/res_sen_roc_del_only.csv'
    finaldf.to_csv(savedfile)

def gen_para(mind, model='para'):
    '''
    generate sen for para dataset
    :param mind:
    :return:
    '''
    test_dataset, df = parse_file_dr(DEV_DR, train_time=False)
    print(df.columns)
    finaldf = eval_model(mind, test_dataset, df, mtd=model)
    savedfile = 'gen_sen/res_sen_para_del_only.csv'
    finaldf.to_csv(savedfile)

def main(ds, mind, para='para'):
    args = {}
    args['n_ctx'] = max_sen_len
    # change to -> load saved dataset
    if ds == 'roc':
        gen_roc(mind, para)
    else:
        gen_para(mind, para)

if __name__ == '__main__':
    ds, mind = sys.argv[1], sys.argv[2]
    if len(sys.argv) == 4:
        main(ds, mind, sys.argv[3])
    else:
        main(ds, mind)
