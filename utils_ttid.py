from dataset import Dataset
import pandas as pd
import sys
from transformers import *
import torch
import numpy as np
import subprocess

sen_text = 'sen_text'
agency = 'agency_cat'
input = 'input'
output = 'out'
POS = '<pos>'
NEG = '<neg>'
EQUAL = '<equal>'
SOURCE = '<source>'
TRAIN_FILE = './data/train_df.xlsx'
DEV_FILE = './data/test_df.xlsx'

max_sen_len = 64
device_ttid = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
token_dict = {
    'bos_token': '<start>',
    'eos_token': '<end>',
    'pad_token': '<pad>',
    'additional_special_tokens': ['<pos>', '<neg>', '<equal>', '<source>']
}
num_added_token = tokenizer.add_special_tokens(token_dict)



def parse_model_inputs(local_labels):
    x = local_labels[0] # b * s
    lens = local_labels[1]
    agens = local_labels[2]
    label = get_label(x, lens)
    tt_ids = get_token_type_ids(x, lens, agens)
    return x, label, tt_ids

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def add_pad(list):
    res = [__sen_pad(sen) for sen in list]
    return res

def __sen_pad(sen):
    # add padding for each sentence
    if len(sen) < max_sen_len:
        pad = [tokenizer.pad_token_id for i in range(max_sen_len - len(sen))]
        sen.extend(pad)
        return sen
    elif len(sen) > max_sen_len:
        orilen = len(sen)
        for i in range(orilen - max_sen_len):
            sen.pop(len(sen) - 2)
    return sen

def get_label(x, lens):
    label = x.clone()
    end_ind = ((x == tokenizer.eos_token_id).nonzero())
    for i in range(x.size()[0]):
        # do not include the last cls token
        startind = lens[i]
        # include the eos token
        endind = end_ind[i][1] + 1
        label[i][0:startind] = torch.FloatTensor([-1 for _ in range(startind)])
        label[i][endind:] = torch.FloatTensor([-1 for _ in range(max_sen_len - endind)])
    return label

def get_token_type_ids(x, lens, agens):
    token_type_ids = np.ones(x.size())
    pad_inds = [(sen == tokenizer.pad_token_id).nonzero() for sen in x]
    gen_time = len(pad_inds) == 1
    if not gen_time:
        pad_inds = [(sen == tokenizer.pad_token_id).nonzero()[0][0] for sen in x]
    for i in range(len(lens)):
        # add SOURCE to ori sen
        token_type_ids[i][0: lens[i]] = [tokenizer.convert_tokens_to_ids(SOURCE) for _ in range(lens[i])]
        if not gen_time:
            # padding for training
            # add agen cat to para sen
            endind = pad_inds[i]
            token_type_ids[i][endind:] = [tokenizer.pad_token_id for _ in range(x.size()[1] - endind)]
        else:
            endind = len(token_type_ids[i])
        token_type_ids[i][lens[i]: endind] = [tokenizer.convert_tokens_to_ids(agens[i]) for _ in range(endind - lens[i])]
    return torch.LongTensor(token_type_ids).to(device)

def make_dataset_para(df, train_time=True):
    df['source_len'] = df['sen0'].str.split().apply(len).tolist()
    if not train_time:
        newdf = pd.DataFrame()
        cats = ['pos', 'equal', 'neg']
        for cat in cats:
            df['des_cat'] = cat
            newdf = newdf.append(df, ignore_index=True)
        df = newdf
        df[input] = '<start> ' + df['sen0']
    else:
        df[input] = '<start> ' + df['sen0'] + ' ' + df['sen1'] + ' <end>'
    list_id = [tokenizer.encode(sen, add_special_tokens=False) for sen in df[input]]
    list_id = add_pad(list_id)
    df['agen_cat1'] = '<' + df['agen_cat1'] + '>'
    dataset = Dataset(list_IDs=list_id, list_len=df['source_len'].tolist(), agen_list=df['agen_cat1'].tolist())
    return dataset, df


def make_dataset(df, train_time=True):
    '''
    :param df: the dataframe
    :param tokenizerparam: the tokenizer
    :param maxlen:
    :param train_time:
    :return: the tokenized dataset to put into data loader
    '''
    df[sen_text] = df[sen_text].astype(str)
    df[sen_text] = df[sen_text].str.strip()
    df[agency] = df[agency].astype(str)
    df[agency] = df[agency].str.strip()
    if not train_time:
        ser = pd.Series()
        ser_ori_cat = pd.Series()
        cats = ['pos', 'equal', 'neg']
        for cat in cats:
            catser = '<start> ' + df[sen_text] + ' <cls> <' + cat + '> '
            ser = ser.append(catser)
            ser_ori_cat = ser_ori_cat.append(df[agency])

        list_in = [tokenizer.encode(sen, add_special_tokens=False) for sen in ser]
        list_in = add_pad(list_in)
        return list_in, ser, ser_ori_cat
    df[input] = '<start> ' + df[sen_text] + ' <cls> <' + df[agency] + '> '
    df[output] = df[sen_text] + ' <end>'
    df[input] = df[input] + df[output]
    list_id = [tokenizer.encode(sen, add_special_tokens=False) for sen in df[input]]
    list_id = add_pad(list_id)
    dataset = Dataset(list_IDs=list_id)
    return dataset
