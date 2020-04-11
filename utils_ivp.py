from utils import *
from dataset_dr import Dataset_dr
import torch
from transformers import *
import sys

sen_text = 'sen'
agency = 'oricat'
input = 'input'
output = 'out'
VER_MAG_RATE = 1.5
VER_ADD_VAL = 5

TRAIN_DR = 'data/parads/train_dr.csv'
DEV_DR = 'data/parads/dev_dr.csv'
TEST_DR = 'data/parads/test_dr.csv'

use_cuda = torch.cuda.is_available()
tokenizer_ivp = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
token_dict = {
    'bos_token': '<start>',
    'eos_token': '<end>',
    'pad_token': '<pad>',
    'cls_token': '<cls>',
    'additional_special_tokens': ['<pos>', '<neg>', '<equal>']
}
num_added_token_ivp = tokenizer_ivp.add_special_tokens(token_dict)
device_ivp = torch.device("cuda:2" if use_cuda else "cpu")


def agen_vector(tokenizer, num_added, multi=True):
    agen_vectors = {}
    for label, verbset in agen_v.items():
        if multi:
            vector = torch.ones(tokenizer.vocab_size + num_added)
        else:
            vector = torch.zeros(tokenizer.vocab_size + num_added)
        for v in verbset:
            forms = infi2allforms(v)
            for form in forms:
                v_li = tokenizer.encode(form)
                if multi:
                    vector[v_li[0]] *= VER_MAG_RATE
                else:
                    vector[v_li[0]] = VER_ADD_VAL
        agen_vectors[label] = vector
    return agen_vectors

def infi2allforms(word):
    res = []
    row = verb_form[verb_form[0] == word]
    if row.empty:
        res.append(word)
        return res
    row = row.dropna(axis=1)
    for col in row.columns:
        res.append(row[col].iloc[0])
    return res

def get_label_ivp(x, batchsize):
    label = x.clone()

    cls_ind = ((x == tokenizer_ivp.cls_token_id).nonzero())
    end_ind = ((x == tokenizer_ivp.eos_token_id).nonzero())
    for i in range(x.size()[0]):
        # do not include the last cls token
        startind = cls_ind[i][1] + 2
        # include the eos token
        endind = end_ind[i][1] + 1
        label[i][0:startind] = torch.FloatTensor([-1 for _ in range(startind)])
        label[i][endind:] = torch.FloatTensor([-1 for _ in range(max_sen_len - endind)])
    return label

def make_dataset(file, maxlen=64, para=True, train_time=True):
    df = pd.read_csv(file)
    if para:
        sen = 'sen0'
        dest = 'sen1'
        agen = 'oricat1'
    else:
        sen = 'sen'
        dest = 'sen'
        agen = 'oricat'
    
    if not train_time:
        ser = pd.Series()
        res_df = pd.DataFrame()
        cats = ['pos', 'equal', 'neg']
        for cat in cats:
            subdf = df.copy()
            catser = '<start> ' + df[sen] + ' <cls> <' + cat + '> '
            ser = ser.append(catser)
            subdf['descat'] = cat
            res_df = res_df.append(subdf)
        list_in = [tokenizer_ivp.encode(s, add_special_tokens=False) for s in ser]
        return list_in, res_df
    df[input] = '<start> ' + df[sen] + ' <cls> <' + df[agen] + \
        '> ' + df[dest] + ' <end>'
    list_id = [tokenizer_ivp.encode(s, add_special_tokens=False) for s in df[input]]
    list_id = add_pad(list_id, tokenizer_ivp)
    dataset = Dataset_dr(list_IDs=list_id)
    return dataset
