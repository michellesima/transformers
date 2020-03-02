from transformers import *
import torch
import pandas as pd
from utils import *
from utils_ivp import *
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from examples.run_generation import *
import sys
from torch import nn
import numpy as np

random_seed = 7
numepoch = 10
softmax = nn.Softmax(dim=0)

verb_stat = {
    'avg': [],
    'std': []
}

#agen_v = agen_vector(tokenizer_ivp, num_added_token_ivp, multi=False)

def repeatN(list, n):
    ori = list
    for _ in range(n):
        list = list.append(ori, ignore_index=True)
    return list

def sample_sequence_ivp(model, length, context, agen_v, num_samples=1, temperature=1, tokenizer=None,top_k=0, top_p=0.0, \
        repetition_penalty=1.0, xlm_lang=None, device='cpu', label='equal', multi=True):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    orilen = len(context)
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.)
            # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)

            verb_vector = agen_v[label]
            verb_idx = verb_vector.nonzero()
            np_vidx = verb_idx.numpy()
            verb_vector = verb_vector.to(device)
            verb_logits = next_token_logits.cpu().numpy()[np_vidx]
            verb_stat['avg'].append(np.average(verb_logits))
            verb_stat['std'].append(np.std(verb_logits))
            if multi:
                next_token_logits = next_token_logits * verb_vector
            else:
                next_token_logits += verb_vector
            for j in set(generated[orilen:].view(-1).tolist()):
                if multi:
                    next_token_logits[j] /= repetition_penalty
                else:
                    next_token_logits[j] -= repetition_penalty
            
            next_token_logits = softmax(next_token_logits)
            
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            #if temperature == 0: #greedy sampling:
            next_token = torch.argmax(filtered_logits).unsqueeze(0)
            #else:
                #next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def gen_p(model, test_dataset, descat):
    outlist = []
    outp = []
    for i in ps:
        for ind in range(len(test_dataset)):
            context_tokens = test_dataset[ind]
            label = descat.iloc[ind]
            out = sample_sequence_ivp(
                model=model,
                agen_v=agen_v,
                context=context_tokens,
                length=max_sen_len,
                top_p=i,
                label=label,
                device=device_ivp
            )
            out = out[0, len(context_tokens):].tolist()
            text = tokenizer_ivp.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)
            end_ind = text.find('<end>')
            if end_ind >= 0:
                text = text[0: end_ind]
            outlist.append(text)
            outp.append(i)
    return outlist, outp

def eval_model(mind, test_dataset, df):
    '''
    get generated sentence for a particular model
    :param mind:
    :param test_dataset:
    :param orisen:
    :param ser_ori_cat:
    :return:
    '''
    modelpath = './modelivp/savedmodels' + str(mind)
    model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
    model.to(device_ivp)
    model.eval()
    outlist, outp = gen_p(model, test_dataset, df['descat'])
    df['out'] = outlist
    df['p-value'] = outp
    return df

def gen_roc(mind):
    '''
    generate sen for roc stories
    :param mind: epoch of model
    :return:
    '''
    test_df = pd.read_csv('./data/roc/dev.csv')
    test_df = test_df.sample(frac=0.1)
    # list of encoded sen
    test_dataset, res_df = make_dataset(test_df, max_sen_len, para=False, train_time=False)
    res_df = repeatN(res_df, len(ps) - 1)
    finaldf = eval_model(mind, test_dataset, res_df) 
    savedfile = 'gen_sen/res_sen_roc_ivp.csv'
    finaldf.to_csv(savedfile)

def gen_para(mind):
    '''
    generate sen for para dataset
    :param mind:
    :return:
    '''
    test_df = pd.read_csv('./data/parads/senp_bs_dev.zip')
    # list of encoded sen
    test_dataset, orisen, ser_ori_cat, ser_para_sen = make_dataset(test_df, max_sen_len, train_time=False)
    orisen = repeatN(orisen, len(ps) - 1)
    ser_ori_cat = repeatN(ser_ori_cat, len(ps) - 1)
    ser_para_sen = repeatN(ser_para_sen, len(ps) - 1)
    finaldf = eval_model(mind, test_dataset, orisen, ser_ori_cat, ser_para_sen)
    savedfile = 'gen_sen/res_sen_para_ivp.xlsx'
    finaldf.to_excel(savedfile)

def main(ds, mind):
    args = {}
    args['n_ctx'] = max_sen_len
    # load saved tokenizer_ivp

    # change to -> load saved dataset
    if ds == 'roc':
        gen_roc(mind)
    else:
        gen_para(mind)


if __name__ == '__main__':
    ds, mind = sys.argv[1], sys.argv[2]
    main(ds, mind)
