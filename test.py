import torch
from transformers import *
import pandas as pd
from torch.utils import data
from dataset import Dataset

sen_text = 'sen_text'
agency = 'agency_cat'
input = 'input'
output = 'out'
random_seed = 7
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


def __make_dataset(df):
    df[input] = '<start> ' + df[sen_text].astype(str) + ' <' + df[agency].astype(str) + '>'
    df[output] = df[sen_text].astype(str) + '<end>'
    tokenizer.bos_token = '<start>'
    tokenizer.eos_token = '<end>'
    tokenizer.pad_token = '<pad>'
    tokenizer.cls_token = ['<pos>', '<neg>', '<equal>']
    max_sen_len = 16
    list_id = [tokenizer.encode(sen, max_length=max_sen_len) for sen in df[input]]
    labels = pd.Series(index=df[input], data=df[output].tolist())
    dataset = Dataset(list_IDs=list_id, labels=labels)
    return dataset

def parse_data():
    data_df = pd.read_excel('data/agency_samples.xlsx')
    newdf = data_df[[sen_text, agency]]
    train_df = newdf.sample(frac=0.8, random_state=random_seed)
    train_dataset = __make_dataset(train_df)
    test_df = pd.concat(newdf, train_df).drop_duplicates(keep=False)
    test_dataset = __make_dataset(test_df)
    return train_dataset, test_dataset

if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    max_epochs = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    train_ds, test_ds = parse_data()
    training_generator = data.DataLoader(train_ds, **params)
    test_generator = data.DataLoader(test_ds, **params)

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)


            # Model computations
            [...]

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in test_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                [...]



