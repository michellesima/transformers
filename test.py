import torch
from transformers import *
import pandas as pd
from torch.utils import data
import torch.optim as optim
from dataset import Dataset
import sys

sen_text = 'sen_text'
agency = 'agency_cat'
input = 'input'
output = 'out'
random_seed = 7
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

'''
1. padding to end
2. mask of loss so that only look at the generation part(excluding the input and padding)
3. input: sen <cls> sen
4. generation file, top k and top p sampling
'''
def __add_pad(list):
    res = [__sen_pad(sen) for sen in list]
    return res

def __sen_pad(sen):
    max_sen_len = 16
    if len(sen) < max_sen_len:
        pad = [tokenizer.pad_token_id for i in range(max_sen_len - len(sen))]
        pad.extend(sen)
        return pad
    elif len(sen) > max_sen_len:
        orilen = len(sen)
        for i in range(orilen - max_sen_len):
            sen.pop(len(sen) - 2)
    return sen

def __make_dataset(df):
    df[input] = '<start> ' + df[sen_text].astype(str) + ' <cls> ' + df[agency].astype(str)
    df[output] = df[sen_text].astype(str) + '<end>'


    list_id = [tokenizer.encode(sen) for sen in df[input]]
    list_id = __add_pad(list_id)
    outsen = [tokenizer.encode(sen) for sen in df[output]]
    outsen = __add_pad(outsen)
    dataset = Dataset(list_IDs=list_id, labels=outsen)
    return dataset

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
    train_dataset = __make_dataset(train_df)
    test_df = newdf[~newdf.isin(train_df)].dropna()
    test_dataset = __make_dataset(test_df)
    return train_dataset, test_dataset

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
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    train_ds, test_ds = parse_data()
    model.resize_token_embeddings(40482)
    training_generator = data.DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
    test_generator = data.DataLoader(test_ds, batch_size=32, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param)
    model.to(device)
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        model.train()
        for local_batch, local_labels in enumerate(training_generator):
            # Transfer to GPUpri
            x = local_labels[0]
            y = local_labels[1]
            outputs = model(x, labels=y)
            loss, logits = outputs[:2]
            loss.backward()
            optimizer.step()
            # Model computations

        # Validation
        model.eval()
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in test_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations



