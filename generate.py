from transformers import *
import torch
import pandas as pd
from utils import *
import torch.nn.functional as F
from examples.run_generation import *
import sys
max_sen_len = 64
random_seed = 7
numepoch = 2

'''
1. 11/11/2019 12:09:44 - WARNING - transformers.tokenization_utils -   This tokenizer does not make use of special tokens. Input is returned with no modification.
a result of calling build_inputs_with_special_tokens, I think legitimate, but not training time
2. mask in test time? 

'''

def sample_seq(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long)
    generated = context
    res = torch.zeros((length, 64), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for i in range(length):
            outputs = model(input_ids=generated[i])  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0] / (temperature if temperature > 0 else 1.)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: #greedy sampling:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            next_token = torch.squeeze(next_token)
            res[i] = next_token
    return res


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    args = {}
    args['n_ctx'] = max_sen_len
    modelpath = 'savedmodels' + str(numepoch)
    model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
    # load saved tokenizer
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    token_dict = {
        'bos_token': '<start>',
        'eos_token': '<end>',
        'pad_token': '<pad>',
        'cls_token': '<cls>'
    }
    num_added_token = tokenizer.add_special_tokens(token_dict)
    # change to -> load saved dataset
    data_df = pd.read_excel('data/agency_samples.xlsx')
    newdf = data_df[[sen_text, agency]]
    train_df = newdf.sample(frac=0.8, random_state=random_seed)
    test_df = newdf[~newdf.isin(train_df)].dropna()
    test_df = test_df.head(2)
    # list of encoded sen
    test_dataset, orisen = make_dataset(test_df, tokenizer, max_sen_len, train_time=False)
    trial = test_dataset
    out = sample_seq(
        model=model,
        context=trial,
        length=len(trial),
        top_p=0.9,
        is_xlnet=False,
        device=device
    )
    outlist = []

    for senind in range(out.size()[0]):
        text = tokenizer.decode(out[senind].tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True)
        outlist.append(text)
    out_ser = pd.Series(data=outlist, dtype=str)
    outdf = pd.DataFrame()
    outdf['ori'] = orisen
    outdf['out'] = outlist
    print(outdf)
    savedfile = 'gen_sen/epoch_tem' + str(numepoch) + '.xlsx'
    outdf.to_excel(savedfile)

if __name__ == '__main__':
    main()