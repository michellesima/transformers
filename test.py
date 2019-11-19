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
numepoch = 3
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
    data_df = pd.read_excel('data/agency_samples.xlsx')
    newdf = data_df[[sen_text, agency]]
    train_df = newdf.sample(frac=0.8, random_state=random_seed)
    train_df.to_excel('./data/train_df.xlsx')
    test_df = newdf[~newdf.isin(train_df)].dropna()
    test_df.to_excel('./data/test_df.xlsx')
    train_dataset = make_dataset(train_df, tokenizer, max_sen_len)
    return train_dataset

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
    print(get_gpu_memory_map())
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    max_epochs = 10
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    pretrained_path = 'savedmodels' + str(numepoch - 1)
    model = OpenAIGPTLMHeadModel.from_pretrained(pretrained_path) #model not on cuda

    train_ds= parse_data()
    training_generator = data.DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param)

    print(get_gpu_memory_map())
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
            os.mkdirs(savepath)
        for local_batch, local_labels in enumerate(training_generator):
            # Transfer to GPUpri
            x = local_labels # b * s
            mask = __get_mask(x)
            outputs = model(x.to(device), attention_mask=mask.to(device))
            x = x[..., 1:] # get rid of start
            logits = outputs[0] # batch size * senlen * v
            logits = logits[..., :-1, :]
            mask = mask[..., 1:]
            logit_view = logits.contiguous().view(batchsize * (max_sen_len - 1), -1)
            mask_view = mask.contiguous().view(-1)
            x_view = x.contiguous().view(-1)
            x_view = x_view.type(torch.FloatTensor)
            masked_x = mask_view * x_view
            for i in range(logit_view.size()[0]):
                logit_view[i] = logit_view[i] * mask_view[i]
            x_view = x_view.type(torch.LongTensor)
            loss = criteria(logit_view.to(device), x_view.to(device))
            losssum += loss
            count += 1
            loss.backward()
            optimizer.step()
            # Model computations
        avg = losssum / count

        model.save_pretrained(savepath)
        losses.append(avg)
    loss_file = 'loss.txt'
    with open(loss_file, 'w') as f:
        f.write(str(losses))





