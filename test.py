import torch
from transformers import *
import pandas as pd
from utils import *
from torch.utils import data
import torch.optim as optim

import sys


random_seed = 7
max_sen_len = 64
batchsize = 4
epoch = 4
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

'''
4. generation file, top k and top p sampling
'''

def parse_data():
    token_dict = {
        'bos_token': '<start>',
        'eos_token': '<end>',
        'pad_token': '<pad>',
        'cls_token': '<cls>'
    }
    num_added_token = tokenizer.add_special_tokens(token_dict)
    data_df = pd.read_excel('data/agency_samples.xlsx')
    newdf = data_df[[sen_text, agency]]
    train_df = newdf.sample(frac=0.8, random_state=random_seed)
    train_dataset = make_dataset(train_df, tokenizer, max_sen_len)
    test_df = newdf[~newdf.isin(train_df)].dropna()
    test_dataset = make_dataset(test_df, tokenizer, max_sen_len)
    return train_dataset, test_dataset

def __get_mask(x):
    mask = torch.zeros((batchsize, max_sen_len))
    cls_ind = ((x == tokenizer.cls_token_id).nonzero())
    end_ind = ((x == tokenizer.eos_token_id).nonzero())
    for i in range(batchsize):
        # do not include the last cls token
        startind = cls_ind[2 * i + 1][1] + 1
        # include the eos token
        endind = end_ind[i][1] + 1
        mask[i][startind: endind] = torch.FloatTensor([1 for j in range(endind - startind)])
    return mask

if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    params = {'batch_size': 64,
              'shuffle': True}
    max_epochs = 1
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    pretrained_path = 'savedmodels' + str(epoch - 1)
    model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path)
    savepath = './savedmodels' + str(epoch)
    train_ds, test_ds = parse_data()
    model.resize_token_embeddings(40482)
    training_generator = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True, pin_memory=True)
    test_generator = data.DataLoader(test_ds, batch_size=32, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param)
    model.to(device)
    ini = 0

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        model.train()
        losssum = 0.0
        count = 0
        for local_batch, local_labels in enumerate(training_generator):
            # Transfer to GPUpri
            x = local_labels[0]
            y = local_labels[1]
            mask = __get_mask(x)
            outputs = model(x, labels=y, attention_mask=mask)
            loss, logits = outputs[:2]
            losssum += loss.item()
            count += 1
            loss.backward()
            optimizer.step()
            # Model computations
        avg = losssum / count
        model.save_pretrained(savepath)
        loss_file = 'loss' + str(epoch) + '.txt'
        with open(loss_file, 'w') as f:
            f.write(str(avg))





