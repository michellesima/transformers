from transformers import *
import torch
import pandas as pd
from utils import *
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from examples.run_generation import *
import sys
max_sen_len = 64
random_seed = 7
numepoch = 10
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")

def repeatN(list, n):
    ori = list
    for _ in range(n):
        list = list.append(ori, ignore_index=True)
    return list


def main():
    args = {}
    args['n_ctx'] = max_sen_len

    # load saved tokenizer
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    token_dict = {
        'bos_token': '<start>',
        'eos_token': '<end>',
        'pad_token': '<pad>',
        'cls_token': '<cls>',
        'additional_special_tokens': ['<pos>', '<neg>', '<equal>']
    }
    num_added_token = tokenizer.add_special_tokens(token_dict)
    # change to -> load saved dataset
    test_df = pd.read_csv('./data/parads/senp_bs_dev.zip')
    test_df = test_df.sample(n=50)
    # list of encoded sen
    test_dataset, orisen, ser_ori_cat = make_dataset_para(test_df, tokenizer, max_sen_len, train_time=False)
    ps = [0.4, 0.6, 0.8, 0.9]
    orisen = repeatN(orisen, len(ps) - 1)
    ser_ori_cat = repeatN(ser_ori_cat, len(ps) - 1)
    finaldf = pd.DataFrame()
    for mind in range(1, 26):
        modelpath = './savedm/savedmodels' + str(mind)
        model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
        model.to(device)
        model.eval()
        outlist = []
        outp = []
        for i in ps:
            for context_tokens in test_dataset:
                out = sample_sequence(
                    model=model,
                    context=context_tokens,
                    length=15,
                    top_p=i,
                    is_xlnet=False,
                    device=device,
                    repetition_penalty = 3
                )
                out = out[0, len(context_tokens):].tolist()
                text = tokenizer.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)
                end_ind = text.find('<end>')
                if end_ind >= 0:
                    text = text[0: end_ind]

                outlist.append(text)
                outp.append(i)

        outdf = pd.DataFrame()

        outdf['ori'] = orisen.tolist()
        outdf['ori_cat'] = ser_ori_cat.tolist()
        outdf['out'] = outlist
        outdf['p-value'] = outp
        outdf = regroup_df(outdf)
        outdf['modelind'] = mind
        finaldf = finaldf.append(outdf, ignore_index=True)
    savedfile = 'gen_sen/res_sen.xlsx'
    finaldf.to_excel(savedfile)

if __name__ == '__main__':
    main()