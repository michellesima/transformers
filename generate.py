from transformers import *
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils import *
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from examples.run_generation import *
import sys

max_sen_len = 20
random_seed = 7
numepoch = 10
ps = [0.4, 0.6, 0.8, 0.9]

def repeatN(list, n):
    ori = list
    for _ in range(n):
        list = list.append(ori, ignore_index=True)
    return list

def sample_sequence(model, length, local_labels, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    xlm_lang=None, device='cpu'):
    # x (b * s)
    context = local_labels[0][0][0: local_labels[1][0].item()]
    context = context.unsqueeze(0)
    generated = context.to(device)
    with torch.no_grad():
        for _ in trange(length):
            tt_ids = get_token_type_ids(generated, local_labels[1], local_labels[2])
            inputs = {'input_ids': generated, 'token_type_ids':tt_ids}
            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.)
            # decrease the prob for period for the first three words
            if _ < 3:
                next_token_logits[1] /= 100
            # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for _ in set(generated.view(-1).tolist()):
                next_token_logits[_] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: #greedy sampling:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def gen_p(model, test_dataset):
    test_generator = DataLoader(test_dataset, batch_size=1, shuffle=True)
    outlist = []
    outp = []
    for i in ps:
        for local_batch, local_labels in enumerate(test_generator):
            out = sample_sequence(
                model=model,
                local_labels=local_labels,
                length=max_sen_len,
                top_p=i,
                device=device
            )
            out = out[0, local_labels[1][0] + 1:].tolist()
            text = tokenizer.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)

            end_ind = text.find('<end>')
            if end_ind >= 0:
                text = text[0: end_ind]
            '''pad_ind = text.find('<pad>')
            if pa
                text = text[0: pad_ind]'''
            outlist.append(text)
            outp.append(i)
    return outlist, outp

def eval_model(mind, test_dataset, df):
    '''
    get generated sentence for a particular model
    '''
    finaldf = pd.DataFrame()
    modelpath = './gen_mo/savedmodels' + str(mind)
    model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
    model.to(device)
    model.eval()
    df = repeatN(df, len(ps) - 1)
    outlist, outp = gen_p(model, test_dataset)
    df['out'] = outlist
    df['p-value'] = outp
    df.sort_values(by=['sen0', 'p-value'], inplace=True)
    df['modelind'] = mind
    finaldf = finaldf.append(df, ignore_index=True)
    return finaldf

def gen_roc(mind):
    '''
    generate sen for roc stories
    :param mind: epoch of model
    :return:
    '''
    test_df = pd.read_excel('./data/dev_df.xlsx')
    # list of encoded sen
    test_dataset, orisen, ser_ori_cat = make_dataset(test_df, train_time=False)
    orisen = repeatN(orisen, len(ps) - 1)
    ser_ori_cat = repeatN(ser_ori_cat, len(ps) - 1)
    finaldf = eval_model(mind, test_dataset, orisen, ser_ori_cat)
    savedfile = 'gen_sen/res_sen.xlsx'
    finaldf.to_excel(savedfile)

def gen_para(mind):
    '''
    generate sen for para dataset
    :param mind:
    :return:
    '''
    test_df = pd.read_csv('./data/parads/senp_bs_dev.zip')
    #test_df = test_df.head(10)
    # list of encoded sen
    test_dataset, test_df = make_dataset_para(test_df, train_time=False)
    finaldf = eval_model(mind, test_dataset, test_df)
    col_names = {
        'agen_cat0': 'ori_cat',
        'agen_cat1': 'para_cat',
        'sen0': 'ori_sen',
        'sen1': 'para_sen'
    }
    finaldf.rename(mapper=col_names, inplace=True, axis=1)
    finaldf = finaldf[['ori_sen', 'ori_cat', 'para_sen', 'para_cat', 'des_cat', 'out', 'p-value']]
    savedfile = 'gen_sen/res_sen.xlsx'
    finaldf.to_excel(savedfile)

def main(ds, mind):
    args = {}
    args['n_ctx'] = max_sen_len
    # load saved tokenizer

    # change to -> load saved dataset
    if ds == 'roc':
        gen_roc(mind)
    else:
        gen_para(mind)




if __name__ == '__main__':
    ds, mind = sys.argv[1], sys.argv[2]
    main(ds, mind)
