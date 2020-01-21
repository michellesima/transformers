import torch
from transformers import *
import pandas as pd
from utils import *
from torch.utils import data
from torch.nn import CrossEntropyLoss
import os.path
from torchsummary import summary
import torch.optim as optim
import sys

random_seed = 7
max_sen_len = 64
batchsize = 4
numepoch = 1
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

'''
problem with training
use label = x, loss = 2, but 
'''

def parse_data():
    token_dict = {
        'bos_token': '<start>',
        'eos_token': '<end>',
        'pad_token': '<pad>',
        'cls_token': '<cls>',
        'additional_special_tokens': ['<pos>', '<neg>', '<equal>']
    }
    num_added_token = tokenizer.add_special_tokens(token_dict)
    dev_df = pd.read_csv('./data/parads/senp_bs_dev.zip')
    dev_dataset,df = make_dataset_para(dev_df)
    return dev_dataset, num_added_token


if __name__ == '__main__':
    # CUDA for PyTorch
    max_epochs = 25
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    dev_ds, num_added= parse_data()
    dev_generator = data.DataLoader(dev_ds, batch_size = batchsize, shuffle=True)
    ini = 0
    train_losses = []
    # Loop over epochs
    for epoch in range(10):
        # Training
        savepath = './savedm/savedmodels' + str(numepoch + epoch)
        if not os.path.exists(savepath):
            break
        model = OpenAIGPTLMHeadModel.from_pretrained(savepath)
        model.to(device)
        model.eval()
        losssum = 0.0
        count = 0
        for local_batch, local_labels in enumerate(dev_generator):
            # Transfer to GPUpri
            x, label, tt_ids = parse_model_inputs(local_labels)
            outputs = model(x.to(device), labels=label.to(device),token_type_ids=tt_ids)
            loss, logits = outputs[:2]
            losssum += loss
            count += 1
            loss.backward()
            # Model computations
        avg = losssum / count
        print(epoch)
        train_losses.append(avg)

    loss_df = pd.DataFrame()
    loss_df["dev_loss"] = train_losses
    loss_df.to_excel('dev_loss.xlsx')





