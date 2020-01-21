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
batchsize = 1
numepoch = 1

def parse_data():

    train_df = pd.read_csv('./data/parads/senp_bs_train.zip')
    train_dataset, df = make_dataset_para(train_df)
    return train_dataset, num_added_token

def train(path='openai-gpt', mind=0):
    max_epochs = 10
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    model = OpenAIGPTLMHeadModel.from_pretrained(path) #model not on cuda
    train_ds, num_added= parse_data()
    if path == 'openai-gpt':
        model.resize_token_embeddings(tokenizer.vocab_size + num_added)
    training_generator = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param, lr=1e-8)
    model.to(device)
    ini = 0
    criteria = CrossEntropyLoss()
    train_losses = []
    # Loop over epochs
    for epoch in range(mind, max_epochs):
        # Training
        savepath = './savedm/savedmodels' + str(epoch + 1)
        model.train()
        losssum = 0.0
        count = 0
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for local_batch, local_labels in enumerate(training_generator):
            # Transfer to GPUpri
            x, label, tt_ids = parse_model_inputs(local_labels)
            outputs = model(x.to(device), labels=label.to(device),token_type_ids=tt_ids)
            loss, logits = outputs[:2]
            losssum += loss
            count += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Model computations
        avg = losssum / count
        print(epoch + 1)
        print(avg)
        print(get_gpu_memory_map())
        model.save_pretrained(savepath)
        train_losses.append(avg)

        loss_df = pd.DataFrame()
        loss_df["train_loss"] = train_losses
        loss_df.to_excel('loss.xlsx')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        train()
    else:
        model = './savedm/savedmodels' + sys.argv[1]
        train(model, int(sys.argv[1]))





