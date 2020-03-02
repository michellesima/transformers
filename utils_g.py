from transformers import *
import pandas as pd
from dataset_g import Dataset_g
from utils import *

batch_g = 4
tokenizer_g = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
special_tok_dict_g = {'pad_token': '<pad>'}
num_added_token_g = tokenizer_g.add_special_tokens(special_tok_dict_g)

def get(row):
    label = np.zeros(3)
    label[0] = row['pos']
    label[1] = row['equal']
    label[2] = row['neg']
    return label

def process_in_g(f):
    df = pd.read_csv(f)
    sen_toks = [tokenizer_g.encode(sen) for sen in df['sen'].tolist()]
    sen_toks = add_pad(sen_toks, tokenizer_g)
    labels = [get_label(row) for idx, row in df.iterrows()]
    dataset = Dataset_g(sen_toks, labels)
