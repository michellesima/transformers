import pandas as pd
from transformers import *

def agen_verbs():
    df = pd.read_csv('~/resources/lexica/CONNOTATION/agency_verb.csv')
    agen_v = {}
    cats = {'+': 'pos', '-':'neg', '=':'equal'}
    for k, v in cats.items():
        subdf = df[df['Agency{agent}_Label'] == k]
        agen_v[v] = set(subdf['verb'].str.split()[0])
    print(agen_v)
    return agen_v

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


