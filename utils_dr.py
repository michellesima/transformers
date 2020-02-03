from dataset import Dataset
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import sys
import csv
import spacy
import codecs
import os
from utils import *
from transformers import *
from dataset_dr import Dataset_dr
import torch
import numpy as np
import subprocess

TRAIN_DR = 'data/parads/train_dr.csv'
DEV_DR = 'data/parads/dev_dr.csv'
TEST_DR = 'data/parads/test_dr.csv'

batchsize_dr = 4
device_dr = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agency_cats = ['<pos>', '<neg>', '<equal>']

tokenizer_dr = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
token_dict_dr = {
    'bos_token': '<start>',
    'eos_token': '<end>',
    'pad_token': '<pad>',
    'sep_token': '<sep>',
    'additional_special_tokens': agency_cats
}
num_added_token_dr = tokenizer_dr.add_special_tokens(token_dict_dr)

def simi_each_word(verb, agenv, vcat, glove_model):
    res = []
    res.append(verb)
    res.append(vcat)
    if verb not in glove_model:
        res.append('none')
        res.append('none')
        res.append('none')
        return res
    for cat in agency_cats:
        cat = cat[1: len(cat)-1]
        verbset = agenv[cat]
        verbset = list(filter(lambda v: v in glove_model, verbset)) 
        if verb in verbset:
            verbset.remove(verb)
        verb_simi = glove_model.most_similar_to_given(verb, verbset)
        res.append(verb_simi)
    return res

def simi_verb_each_cat():
    '''
    for each word, get its most similar word in each cat from glove emb
    both word and its simi words are supposed to be in infinitive form
    but 721 words cannot get infi form
    '''
    # cat -> infi form of words
    agenv = agen_verbs()
    df = pd.DataFrame()
    glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
    vs_col = ['verb', 'oricat']
    vs_col.extend(agency_cats)
    for cat, verbset in agenv.items():
        data = [simi_each_word(v, agenv, cat, glove_model) for v in verbset]
        catdf = pd.DataFrame(data, columns=vs_col)
        df = df.append(catdf)
    df.to_csv('verb2simi.csv')


def sen_in(sen, train_time=True):
    sen = sen[1]
    sen['sen'] = sen['sen'].lower()
    sen_li = sen['sen'].split()
    verb_idx = sen['verb_idx'].split()
    verb_len = sen['verb_len'].split()
    verbs = ''
    for i in range(len(verb_idx)):
        idx = int(verb_idx[i])
        num = int(verb_len[i])
        for j in range(num):
            verbs += sen_li[idx] + ' '
            del sen_li[idx]
    newsen = '<start> '
    for w in sen_li:
        newsen += w + ' '
    if not train_time:
        newsen = newsen + '<sep> ' + verbs + '<start>'
    else:
        newsen += '<sep> ' + verbs + '<start> ' + sen['sen'] + ' <end>'
    tok_li = tokenizer_dr.encode(newsen, add_special_tokens=False)
    return tok_li

def sen_in_retr(sen, df, method):
    senavg = df[df['sen']==sen]['glove_avg']
    df['glove_avg'] = df['glove_avg'] - senavg


def parse_file_dr(file, train_time=True, head=None, method=None):
    path = os.path.abspath(file)
    with open(path,encoding='UTF-8') as f:
        df = pd.read_csv(f)
        if not head is None:
            df = df.sample(n=head)
        tok_li = [sen_in(sen, train_time=train_time) for sen in df.iterrows()]
        if not train_time:
            return tok_li, df
        tok_li = add_pad(tok_li)
        dataset = Dataset_dr(list_IDs=tok_li)
        return dataset

def get_label_dr(x):
    label = x.clone()
    start_inds = ((x == tokenizer.bos_token_id).nonzero())
    end_inds = ((x == tokenizer.eos_token_id).nonzero())
    for i in range(x.size()[0]):
        # do not include the last cls token
        startind = start_inds[2*i+1][1].item() + 1
        endind = end_inds[i][1].item() + 1
        # do not include second start
        label[i][0:startind] = torch.FloatTensor([-1 for _ in range(startind)])
        # include end token
        label[i][endind:] = torch.FloatTensor([-1 for _ in range(max_sen_len - endind)])
    return label

def parse_model_inputs_dr(local_labels):
    x = local_labels # b * s
    label = get_label_dr(x)
    return x, label
