from transformers import *
import torch
from generate_ivp import sample_sequence_ivp
from utils_ivp import agen_vector
from utils_dr import *
import sys


if __name__ == '__main__':
    sen, label, mtd, mind, repe = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    agenv = agen_vector(tokenizer_dr, num_added_token_dr, multi=False)
    sen = '<start> ' + sen + ' <cls> ' + label + ' <start>' 
    sen = tokenizer_dr.encode(sen)
    senlen = len(sen)
    if mtd == 'para':
        savedir = './modelp/savedmodels'
    elif mtd == 'mix':
        savedir = './modelmix/savedmodels'
    else:
        savedir = './modelr/savedmodels'
    modelpath = savedir + str(mind)
    print(modelpath)
    model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
    model.to(device_dr)
    model.eval()
    out = sample_sequence_ivp(
        model=model,
        tokenizer=tokenizer_dr,
        context=sen,
        agen_v=agenv,
        length=20,
        top_p=0.4,
        repetition_penalty=float(repe),
        label=label,
        multi=False,
        device=device_dr
    )
    out = out[0, senlen:].tolist()
    text = tokenizer_dr.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)
    end_ind = text.find('<end>')
    if end_ind >= 0:
        text = text[0: end_ind]
    print(text)

