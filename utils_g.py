from transformers import *
import torch
import pandas as pd
import numpy as np
import pickle
from dataset_g import Dataset_g
from utils import *
import os

DIRG =  'data/roc/forg/'


TRAIN_G = DIRG + 'train.pickle'
DEV_G = DIRG + 'dev.pickle'
TEST_G = DIRG + 'test.pickle'

PRELOADED_TRAIN_G = DIRG + 'train_pre.pickle'
PRELOADED_DEV_G = DIRG + 'dev_pre.pickle'
PRELOADED_TEST_G = DIRG + 'test_pre.pickle'

path_match = {
    TRAIN_G: PRELOADED_TRAIN_G,
    DEV_G: PRELOADED_DEV_G,
    TEST_G:PRELOADED_TEST_G
}

batch_g = 4
tokenizer_g = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
special_tok_dict_g = {'pad_token': '<pad>',
    'sep_token': '<sep>',
    'bos_token': '<start>',
    'eos_token': '<end>',
    'additional_special_tokens': ['<VERB>']
}
num_added_token_g = tokenizer_g.add_special_tokens(special_tok_dict_g)
v_toks = tokenizer_g.encode('<VERB>')
print(v_toks)
device_g = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def get(row):
    label = np.zeros(3)
    label[0] = row['pos']
    label[1] = row['equal']
    label[2] = row['neg']
    return label

def process_seng(row):
    '''
    [sen, sendel, cat, e]
    '''
    tsen_li = []
    tsendel_li = []
    row[0] = '<start> ' + row[0] + '<end>'
    row[1] = '<start> ' + row[1] + '<end>'
    sen_li = row[0].split()
    sendel_li = row[1].split()
    e = row[3]
    emptycat = np.zeros((1,3))
    # add e for start
    e = np.append(emptycat, e, axis=0)
    # add e for end
    e = np.append(e, emptycat, axis=0)
    esen = np.zeros((1,3))
    edelsen = np.zeros((1,3))
    for i in range(len(sen_li)):
        word = sen_li[i]
        toks = tokenizer_g.encode(word)
        tsen_li.extend(toks)
        toapp = np.expand_dims(e[i], axis=0)
        if len(toks) > 1:
            toapp = np.repeat(toapp, len(toks), axis=0)
        print(esen.shape, ' ', toapp.shape)
        esen = np.append(esen, toapp, axis=0)
        if sendel_li[i] == '<VERB>':
            tsendel_li.extend(v_toks)
            edelsen = np.append(edelsen, np.expand_dims(e[i], axis=0), axis=0)
        else:
            tsendel_li.extend(toks)
            edelsen = np.append(edelsen, toapp, axis=0)
    esen = esen[1:]
    edelsen = edelsen[1:]
    tsen_li.append(tokenizer_g.sep_token_id)
    label_li = [-1 for _ in range(len(tsen_li))]
    label_li.extend(tsendel_li)
    label_pads = [-1 for _ in range(max_sen_len-len(label_li))]
    label_li.extend(label_pads)
    tsen_li.extend(tsendel_li)
    esen = np.append(esen, emptycat, axis=0)
    esen = np.append(esen, edelsen, axis=0)
    lenpad = max_sen_len - len(tsen_li)
    esen = np.append(esen, np.repeat(emptycat, lenpad, 0), axis=0)
    return tsen_li, esen, label_li

def process_in_g(f):
    if not os.path.exists(path_match[f]):
        data = pickle.load(open(f, 'rb'))
        print(len(data))
        tem = [process_seng(row) for row in data]
        tem = np.array(tem)
        pickle.dump(tem, open(path_match[f], 'wb'))
    else:
        tem = pickle.load(open(path_match[f], 'rb'))
    toks_li = tem[:, 0]
    padded_es = tem[:, 1]
    padded_labels = tem[:, 2]
    toks_li = add_pad(toks_li, tokenizer_g)
    dataset = Dataset_g(toks_li, padded_es, padded_labels)
    return dataset

