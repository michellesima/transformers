import pandas as pd
from nltk.stem import WordNetLemmatizer 
from transformers import *

max_sen_len = 64
lemmatizer = WordNetLemmatizer() 

ROC_TRAIN = './data/roc/train.csv'
ROC_TEST = './data/roc/test.csv'
ROC_DEV = './data/roc/dev.csv'

def agen_verbs():
    '''
    for word in each category, get its infinitive form if it's in en
    try if it's in glove, then get the most similar word from glove and store it
    in the data file
    Note: 721 words will not be changed
    '''
    df = pd.read_csv('~/resources/lexica/CONNOTATION/agency_verb.csv')
    agen_v = {}
    cats = {'+': 'pos', '-':'neg', '=':'equal'}
    for k, v in cats.items():
        subdf = df[df['Agency{agent}_Label'] == k]
        ver_li = subdf['verb'].str.split()
        agen_v[v] = set(word_infinitive(li[0]) for li in ver_li if len(li) > 0)
    return agen_v

def word_infinitive(word):
    infi = lemmatizer.lemmatize(word)
    return infi 

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

def add_pad(list, tokenizer):
    res = [__sen_pad(sen, tokenizer) for sen in list]
    return res

def __sen_pad(sen, tokenizer):
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


