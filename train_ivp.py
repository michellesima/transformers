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
numepoch = 1

def parse_data():
    train_df = pd.read_csv('~/transformers/data/parads/senp_bs_train.zip')
    train_dataset = make_dataset_para(train_df, tokenizer_ivp, max_sen_len)
    return train_dataset, num_added_token


if __name__ == '__main__':
    # CUDA for PyTorch
    max_epochs = 10
    # Load dataset, tokenizer_ivp, model from pretrained model/vocabulary
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt') #model not on cuda
    train_ds, num_added= parse_data()
    model.resize_token_embeddings(tokenizer_ivp.vocab_size + num_added)
    training_generator = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param, lr=5e-6)
    model.to(device_ivp)
    ini = 0
    criteria = CrossEntropyLoss()
    train_losses = []
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
            x = local_labels # b * s
            label = get_label(x, batchsize)
            outputs = model(x.to(device_ivp), labels=label.to(device_ivp))
            loss, logits = outputs[:2]
            losssum += loss
            count += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Model computations
        avg = losssum / count
        print(epoch)
        model.save_pretrained(savepath)
        train_losses.append(avg)

    loss_df = pd.DataFrame()
    loss_df["train_loss"] = train_losses
    loss_df.to_excel('loss.xlsx')





