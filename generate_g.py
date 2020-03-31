import sys
from transformers.modeling_openai import OpenAIGPTLMHeadAgenModel
import numpy as np
from transformers import *
import torch
from torch.utils.data import DataLoader
from generate_ivp import sample_sequence_ivp
import pandas as pd
from utils import *
from utils_g import *
from utils_ivp import agen_vector
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from examples.run_generation import *

ps = [0.4, 0.6]
cats = {
    'pos': 0,
    'equal': 1,
    'neg': 2
}

def gen_p(model, test_dataset, descat, cat_head):
    outlist = []
    outp = []
    sens = []
    for i in ps:
        for j in range(len(test_dataset)):
            sen = test_dataset[j]
            sepind = sen.index(40481)
            sen = sen[:sepind+2]
            senlen = len(sen)
            e = torch.FloatTensor(descat[j]).to(device_g)
            # e [1,3]
            out = sample_sequence(
                model=model,
                context=sen,
                length=max_sen_len,
                top_p=i,
                e=e,
                cat_head=cat_head,
                device=device_g
            )
            out = out[0, senlen:].tolist()
            text = tokenizer_g.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)
            sen = tokenizer_g.decode(sen, clean_up_tokenization_spaces=True, skip_special_tokens=False)
            end_ind = text.find('<')
            if end_ind >= 0:
                text = text[0: end_ind]
            sens.append(sen)
            outlist.append(text)
            outp.append(i)
    return outlist, outp, sens

def add_cat(dataset):
    resds = []
    descat = []
    es = np.zeros((1, 3))
    es[0][0] = 1
    descat = ['pos'] * len(dataset)
    es = np.repeat(es, len(dataset), axis=0)
    eq = np.zeros((1, 3))
    eq[0][1] = 1
    tem = ['equal'] * len(dataset)
    descat.extend(tem)
    eqs = np.repeat(eq, len(dataset), axis=0)
    es = np.append(es, eqs, axis=0)
    en = np.zeros((1, 3))
    en[0][2] = 1
    tem = ['neg'] * len(dataset)
    descat.extend(tem)
    ens = np.repeat(en, len(dataset), axis=0)
    es = np.append(es, ens, axis=0)
    for i in range(3):
        resds.extend(dataset)
    print(len(dataset))
    return resds, es, descat

def gen_roc(model, cat_head):
    test_dataset = process_in_g(ROC_DEV_G, train=False)
    test_dataset, es, descat = add_cat(test_dataset)
    outlist, outp, sens = gen_p(model, test_dataset, es, cat_head)
    df = pd.DataFrame()
    test_dataset = pd.Series(data=test_dataset)
    descat = pd.Series(data=descat)
    df['sen'] = sens
    df['descat'] = repeatN(descat, len(ps)-1)
    print(len(df.index))
    print(len(outp))
    df['p-value'] = outp
    df['out'] = outlist
    df.to_csv('./gen_sen/res_sen_roc_g.csv')


def main(ds, mind):
    savepath = './modelgp/savedmodels' + mind

    modelg = OpenAIGPTLMHeadAgenModel.from_pretrained('openai-gpt')
    modelg.resize_token_embeddings(tokenizer_g.vocab_size + num_added_token_g)
    path = os.path.join(savepath, 'model.bin')
    modelg.load_state_dict(torch.load(path))
    modelg.to(device_g)
    model = OpenAIGPTLMHeadModel.from_pretrained('./modelmix/savedmodels3')
    model.to(device_g)
    gen_roc(model, modelg.cat_head)

if __name__ == '__main__':
    ds, mind = sys.argv[1], sys.argv[2]
    main(ds, mind)
