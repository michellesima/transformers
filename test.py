import torch
from transformers import *
import pandas as pd
from utils import *
from torch.utils import data
from torch.nn import CrossEntropyLoss
import os
from torchsummary import summary
import torch.optim as optim
import sys

random_seed = 7
max_sen_len = 64
batchsize = 4
numepoch = 2
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')



def parse_data():
    token_dict = {
        'bos_token': '<start>',
        'eos_token': '<end>',
        'pad_token': '<pad>',
        'cls_token': '<cls>',
        'additional_special_tokens': ['<pos>', '<neg>', '<equal>']
    }
    num_added_token = tokenizer.add_special_tokens(token_dict)
    train_df = pd.read_excel('./data/train_df.xlsx')
    train_dataset = make_dataset(train_df, tokenizer, max_sen_len)
    return train_dataset, num_added_token


if __name__ == '__main__':
    # CUDA for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = 5
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    pretrained_path = './savedm/savedmodels' + str(numepoch - 1)
    model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path) #model not on cuda
    train_ds, num_added= parse_data()
    model.resize_token_embeddings(tokenizer.vocab_size + num_added)
    training_generator = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param)
    model.to(device)
    ini = 0
    criteria = CrossEntropyLoss()
    losses = []
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        savepath = './savedm/savedmodels' + str(numepoch + epoch)
        model.train()
        losssum = 0.0
        count = 0
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for local_batch, local_labels in enumerate(training_generator):
            # Transfer to GPUpri
            optimizer.zero_grad()
            x = local_labels # b * s
            mask = get_mask(x, batchsize)
            outputs = model(x.to(device), attention_mask=mask.to(device))
            logit_view, x_view = prepare_loss(x, outputs, mask, batchsize)
            loss = criteria(logit_view.to(device), x_view.to(device))
            losssum += loss
            count += 1
            loss.backward()
            optimizer.step()
            # Model computations
        avg = losssum / count
        print(epoch)
        model.save_pretrained(savepath)
        losses.append(avg)
    loss_file = 'loss.txt'
    with open(loss_file, 'w') as f:
        f.write(str(losses))





