from dataset import Dataset
import pandas as pd
import sys
import subprocess

sen_text = 'sen_text'
agency = 'agency_cat'
input = 'input'
output = 'out'
max_sen_len = 64
tokenizer = None


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

def __add_pad(list):
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


def make_dataset(df, tokenizerparam, maxlen, train_time=True):
    '''
    :param df: the dataframe
    :param tokenizerparam: the tokenizer
    :param maxlen:
    :param train_time:
    :return: the tokenized dataset to put into data loader
    '''
    global tokenizer
    tokenizer = tokenizerparam
    max_sen_len = maxlen
    df[sen_text] = df[sen_text].astype(str)
    df[sen_text] = df[sen_text].str.strip()
    df[agency] = df[agency].astype(str)
    df[agency] = df[agency].str.strip()
    if not train_time:
        ser = pd.Series()
        cats = ['pos', 'equal', 'neg']
        for cat in cats:
            catser = '<start> ' + df[sen_text] + ' <cls> <' + cat + '> <cls> '
            ser = ser.append(catser)

        list_in = [tokenizer.encode(sen, add_special_tokens=False) for sen in ser]
        list_in = __add_pad(list_in)
        return list_in, ser
    df[input] = '<start> ' + df[sen_text] + ' <cls> ' + df[agency] + ' <cls> '
    df[output] = df[sen_text] + ' <end>'
    df[input] = df[input] + df[output]
    list_id = [tokenizer.encode(sen, add_special_tokens=False) for sen in df[input]]
    list_id = __add_pad(list_id)
    dataset = Dataset(list_IDs=list_id)
    return dataset