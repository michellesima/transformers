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
    test_df = pd.read_excel('./data/test_df.xlsx')
    test_df = test_df.head(10)
    # list of encoded sen
    test_dataset, orisen, ser_ori_cat = make_dataset(test_df, tokenizer, max_sen_len, train_time=False)
    for i in range(numepoch):
        modelpath = './savedm/savedmodels' + str(i + 1)
        model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
        model.to(device)
        outlist = []
        for context_tokens in test_dataset:
            out = sample_sequence(
                model=model,
                context=context_tokens,
                length=20,
                top_p=0.9,
                is_xlnet=False,
                device=device
            )
            out = out[0, len(context_tokens):].tolist()
            text = tokenizer.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            outlist.append(text)

        out_ser = pd.Series(data=outlist, dtype=str)
        outdf = pd.DataFrame()
        outdf['ori'] = orisen
        outdf['ori_cat'] = ser_ori_cat
        outdf['out'] = outlist
        outdf = regroup_df(outdf)
        savedfile = 'gen_sen/epoch_tem' + str(i + 1) + '.xlsx'
        outdf.to_excel(savedfile)

if __name__ == '__main__':
    main()