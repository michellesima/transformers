
from transformers.modeling_openai import OpenAIGPTLMHeadAgenModel
import torch
from transformers import *
import pandas as pd
from utils import *
from utils_g import *
from torch.utils import data
from torch.nn import CrossEntropyLoss
import os.path
from torchsummary import summary
import torch.optim as optim
import sys

random_seed = 7
max_sen_len = 64
batchsize = 4

if __name__ == '__main__':
    ds = sys.argv[1]
    # CUDA for PyTorch
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    if ds == 'roc':
        dev_ds = process_in_g(ROC_DEV_G)
        savedir = './modelg/savedmodels'
    elif ds == 'para':
        dev_ds = process_in_g(PARA_DEV_G)
        savedir = './modelgp/savedmodels'
    else:
        dev_ds = process_in_g(PARA_DEV_G)
        dev_roc = process_in_g(ROC_DEV_G)
        dev_ds.append(train_roc)
        savedir = './modelgm/savedmodels'
    savedir = './modelg/savedmodels'
    dev_generator = data.DataLoader(dev_ds, batch_size = batchsize, shuffle=True)
    ini = 0
    train_losses = []
    # Loop over epochs
    for epoch in range(10):
        # Training
        savepath = savedir + str(epoch + 1)
        if not os.path.exists(savepath):
            break
        model = OpenAIGPTLMHeadAgenModel.from_pretrained('openai-gpt')
        model.resize_token_embeddings(tokenizer_g.vocab_size + num_added_token_g)
        path = os.path.join(savepath, 'model.bin')
        model.load_state_dict(torch.load(path))
        model.to(device_g)
        model.eval()
        losssum = 0.0
        count = 0
        for local_batch, local_labels in enumerate(dev_generator):
            #print(local_labels)
            # Transfer to GPUpri
            x = local_labels[0].to(device_g)
            e = local_labels[1].to(device_g)
            outputs = model(x, e=e, labels=x)
            loss, logits = outputs[:2]
            losssum += loss
            count += 1
            loss.backward()
            # Model computations
        avg = losssum / count
        print(epoch)
        print(avg)
        train_losses.append(avg)

    loss_df = pd.DataFrame()
    loss_df["dev_loss"] = train_losses
    loss_df.to_csv('dev_loss_g.csv')





