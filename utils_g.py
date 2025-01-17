from transformers import *
import torch
import pandas as pd
import numpy as np
import pickle
from dataset_g import Dataset_g
from utils import *
import os

ROC_DIRG =  'data/roc/forg/'


ROC_TRAIN_G = ROC_DIRG + 'train.pickle'
ROC_DEV_G = ROC_DIRG + 'dev.pickle'
ROC_TEST_G = ROC_DIRG + 'test.pickle'

ROC_PRELOADED_TRAIN_G = ROC_DIRG + 'train_pre.pickle'
ROC_PRELOADED_DEV_G = ROC_DIRG + 'dev_pre.pickle'
ROC_PRELOADED_TEST_G = ROC_DIRG + 'test_pre.pickle'

PARA_DIRG = 'data/parads/forg'

PARA_TRAIN_G = os.path.join(PARA_DIRG, 'para_train.pickle')
PARA_DEV_G = os.path.join(PARA_DIRG, 'para_dev.pickle')
PARA_TEST_G = os.path.join(PARA_DIRG, 'para_test.pickle')


PARA_PRELOADED_TRAIN_G = os.path.join(PARA_DIRG, 'train_pre.pickle')
PARA_PRELOADED_DEV_G = os.path.join(PARA_DIRG, 'dev_pre.pickle')
PARA_PRELOADED_TEST_G = os.path.join(PARA_DIRG, 'test_pre.pickle')
path_match = {
    ROC_TRAIN_G: ROC_PRELOADED_TRAIN_G,
    ROC_DEV_G: ROC_PRELOADED_DEV_G,
    ROC_TEST_G:ROC_PRELOADED_TEST_G,
    PARA_TRAIN_G: PARA_PRELOADED_TRAIN_G,
    PARA_DEV_G: PARA_PRELOADED_DEV_G,
    PARA_TEST_G: PARA_PRELOADED_TEST_G

}

batch_g = 4
tokenizer_g = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
special_tok_dict_g = {'pad_token': '<pad>',
    'bos_token': '<start>',
    'eos_token': '<end>',
    'pad_token': '<pad>',
    'cls_token': '<cls>',
    'additional_special_tokens': ['<pos>', '<neg>', '<equal>', '<VERB>']
}
num_added_token_g = tokenizer_g.add_special_tokens(special_tok_dict_g)
v_toks = tokenizer_g.encode('<VERB>')
device_g = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

t = 0
'''
def get(row):    
	label = np.zeros(3)
    label[0] = row['pos']
    label[1] = row['equal']
    label[2] = row['neg']
    return label
'''

def get_label_g(x):
    label = x.clone()
    pad_inds = x == tokenizer_g.pad_token_id
    label[pad_inds] = -1
    return label

def process_seng(row, train=True):
    '''
    [sen, sendel, cat, e]
    '''
    tsen_li = []
    tsendel_li = []
    if row[1].find('<VERB>') == -1:
        global t
        t += 1
        print(row[0])
        print(row[1])
    sen_li = row[0].split()
    sendel_li = row[1].split()
    e = row[3]
    emptycat = np.zeros((1,3))
    esen = np.zeros((1,3))
    edelsen = np.zeros((1,3))
    for i in range(len(sen_li)):
        word = sen_li[i]
        toks = tokenizer_g.encode(word)
        tsen_li.extend(toks)
        toapp = np.expand_dims(e[i], axis=0)
        if len(toks) > 1:
            toapp = np.repeat(toapp, len(toks), axis=0)
        esen = np.append(esen, toapp, axis=0)
        if sendel_li[i] == '<VERB>':
            tsendel_li.extend(v_toks)
            edelsen = np.append(edelsen, np.expand_dims(e[i], axis=0), axis=0)
        else:
            tsendel_li.extend(toks)
            edelsen = np.append(edelsen, toapp, axis=0)
    edelsen = edelsen[1:]
    lenpad = max_sen_len - esen.shape[0]
    esen = np.append(esen, np.repeat(emptycat, lenpad, 0), axis=0)
    return tsen_li, esen

def process_seng_dev(row, train=True):
    '''
    [sen, sendel, cat, e]
    '''
    sendel = row[1]
    sendel = '<start> ' + sendel
    sendel = sendel + ' <end>'
    sen_li = tokenizer_g.encode(sendel)
    return row[0], sen_li
    '''
    tsen_li = []
    tsen_li.append(tokenizer_g.bos_token_id)
    tsendel_li = []
    tsendel_li.append(tokenizer_g.bos_token_id)
    sen_li = row[0].split()
    sendel_li = row[1].split()
    cat = '<' + row[2] + '>'
    e = row[3]
    emptycat = np.zeros((1,3))
    esen = np.zeros((1,3))
    edelsen = np.zeros((1,3))
    for i in range(len(sen_li)):
        word = sen_li[i]
        toks = tokenizer_g.encode(word)
        tsen_li.extend(toks)
        toapp = np.expand_dims(e[i], axis=0)
        if len(toks) > 1:
            toapp = np.repeat(toapp, len(toks), axis=0)
        esen = np.append(esen, toapp, axis=0)
        if sendel_li[i] == '<VERB>':
            tsendel_li.extend(v_toks)
            edelsen = np.append(edelsen, np.expand_dims(e[i], axis=0), axis=0)
        else:
            tsendel_li.extend(toks)
            edelsen = np.append(edelsen, toapp, axis=0)
    # eos token 
    tsen_li.append(tokenizer_g.eos_token_id)
    esen = np.append(esen, emptycat, axis=0)
    tsendel_li.append(tokenizer_g.eos_token_id)
    edelsen = np.append(edelsen, emptycat, axis=0)
    # cat 
    tsendel_li.extend(tokenizer_g.encode(cat))
    edelsen = np.append(edelsen, emptycat, axis=0)
    edelsen = np.append(edelsen, esen, axis=0)
    tsendel_li.extend(tsen_li)

    lenpad = max_sen_len - len(tsendel_li)
    edelsen = np.append(edelsen, np.repeat(emptycat, lenpad, 0), axis=0)
    return tsendel_li, edelsen
    '''

def process_in_g(f, train=True):
    if not train:
        data = pickle.load(open(f, 'rb'))
        tem = [process_seng_dev(row) for row in data]
        tem = tem[:100]
        sen = [row[0] for row in tem]
        sendel = [row[1] for row in tem]
        return sen, sendel
    if not os.path.exists(path_match[f]):
        data = pickle.load(open(f, 'rb'))
        tem = [process_seng(row) for row in data]
        tem = np.array(tem)
        pickle.dump(tem, open(path_match[f], 'wb'))
    else:
        tem = pickle.load(open(path_match[f], 'rb'))
    toks_li = tem[:, 0]
    padded_es = tem[:, 1]
    toks_li = add_pad(toks_li, tokenizer_g)
    print(len(toks_li))
    dataset = Dataset_g(toks_li, padded_es)
    return dataset

