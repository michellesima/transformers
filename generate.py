from transformers import *
import torch
import pandas as pd
from utils import *
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from examples.run_generation import *
import sys
max_sen_len = 20
random_seed = 7
numepoch = 10
ps = [0.4, 0.6, 0.8, 0.9]
use_cuda = torch.cuda.is_available()
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
token_dict = {
    'bos_token': '<start>',
    'eos_token': '<end>',
    'pad_token': '<pad>',
    'cls_token': '<cls>',
    'additional_special_tokens': ['<pos>', '<neg>', '<equal>']
}
num_added_token = tokenizer.add_special_tokens(token_dict)
device = torch.device("cuda:2" if use_cuda else "cpu")

def repeatN(list, n):
    ori = list
    for _ in range(n):
        list = list.append(ori, ignore_index=True)
    return list

def gen_p(model, test_dataset):
    outlist = []
    outp = []
    for i in ps:
        for context_tokens in test_dataset:
            out = sample_sequence(
                model=model,
                context=context_tokens,
                length=max_sen_len,
                top_p=i,
                is_xlnet=False,
                device=device
            )
            out = out[0, len(context_tokens):].tolist()
            text = tokenizer.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)
            end_ind = text.find('<end>')
            if end_ind >= 0:
                text = text[0: end_ind]

            outlist.append(text)
            outp.append(i)
    return outlist, outp

def eval_model(mind, test_dataset, orisen, ser_ori_cat, ser_para = pd.Series()):
    '''
    get generated sentence for a particular model
    :param mind:
    :param test_dataset:
    :param orisen:
    :param ser_ori_cat:
    :return:
    '''
    finaldf = pd.DataFrame()
    modelpath = './savedm/savedmodels' + str(mind)
    model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
    model.to(device)
    model.eval()
    outlist, outp = gen_p(model, test_dataset)
    outdf = pd.DataFrame()
    outdf['ori'] = orisen.tolist()
    outdf['ori_cat'] = ser_ori_cat.tolist()
    if ser_para.size> 0:
        outdf['para_sen'] = ser_para.tolist()
    outdf['out'] = outlist
    outdf['p-value'] = outp
    outdf = regroup_df(outdf)
    outdf['modelind'] = mind
    finaldf = finaldf.append(outdf, ignore_index=True)
    return finaldf

def gen_roc(mind):
    '''
    generate sen for roc stories
    :param mind: epoch of model
    :return:
    '''
    test_df = pd.read_excel('./data/dev_df.xlsx')
    # list of encoded sen
    test_dataset, orisen, ser_ori_cat = make_dataset(test_df, tokenizer, max_sen_len, train_time=False)
    orisen = repeatN(orisen, len(ps) - 1)
    ser_ori_cat = repeatN(ser_ori_cat, len(ps) - 1)
    finaldf = eval_model(mind, test_dataset, orisen, ser_ori_cat)
    savedfile = 'gen_sen/res_sen.xlsx'
    finaldf.to_excel(savedfile)

def gen_para(mind):
    '''
    generate sen for para dataset
    :param mind:
    :return:
    '''
    test_df = pd.read_csv('./data/parads/senp_bs_dev.zip')
    # list of encoded sen
    test_dataset, orisen, ser_ori_cat, ser_para_sen = make_dataset_para(test_df, tokenizer, max_sen_len, train_time=False)
    orisen = repeatN(orisen, len(ps) - 1)
    ser_ori_cat = repeatN(ser_ori_cat, len(ps) - 1)
    ser_para_sen = repeatN(ser_para_sen, len(ps) - 1)
    finaldf = eval_model(mind, test_dataset, orisen, ser_ori_cat, ser_para_sen)
    savedfile = 'gen_sen/res_sen.xlsx'
    finaldf.to_excel(savedfile)

def main(ds, mind):
    args = {}
    args['n_ctx'] = max_sen_len
    # load saved tokenizer

    # change to -> load saved dataset
    if ds == 'roc':
        gen_roc(mind)
    else:
        gen_para(mind)


if __name__ == '__main__':
    ds, mind = sys.argv[1], sys.argv[2]
    main(ds, mind)