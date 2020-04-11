import torch
from transformers import *
import pandas as pd
from utils import *
from utils_ivp import *
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
    train_dataset = make_dataset(TRAIN_DR)
    #roc_ds = make_dataset(ROC_TRAIN, para=False)
    #train_dataset.append(roc_ds)
    return train_dataset


if __name__ == '__main__':
    # CUDA for PyTorch
    max_epochs = 10
    # Load dataset, tokenizer_ivp, model from pretrained model/vocabulary
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt') #model not on cuda
    train_ds = parse_data()
    model.resize_token_embeddings(tokenizer_ivp.vocab_size + num_added_token_ivp)
    training_generator = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param, lr=1e-5)
    model.to(device_ivp)
    ini = 0
    criteria = CrossEntropyLoss()
    train_losses = []
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        savepath = './modelivp/savedmodels' + str(numepoch + epoch)
        model.train()
        losssum = 0.0
        count = 0
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for local_batch, local_labels in enumerate(training_generator):
            # Transfer to GPUpri
            x = local_labels # b * s
            label = get_label_ivp(x, batchsize)
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
        print(avg)
        model.save_pretrained(savepath)
        train_losses.append(avg)

    loss_df = pd.DataFrame()
    loss_df["train_loss"] = train_losses
    loss_df.to_csv('loss.csv')





