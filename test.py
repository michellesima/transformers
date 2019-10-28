import torch
from transformers import *
import pandas as pd
from data import Dataset

sen_text = 'sen_text'
agency = 'agency_cat'
input = 'input'
output = 'out'
random_seed = 7

def __make_dataset(df):
    df[input] = '<start> ' + df[sen_text].astype(str) + ' <' + df[agency].astype(str) + '>'
    df[output] = df[sen_text].astype(str) + '<end>'
    list_id = df[input].tolist()
    labels = pd.Series(index=df[input], data=df[output].tolist())
    dataset = Dataset(list_IDs=list_id, labels=labels)
    return dataset

def parse_data():
    data_df = pd.read_excel('data/agency_samples.xlsx')
    newdf = data_df[[sen_text, agency]]
    train_df = newdf.sample(frac=0.8, random_state=random_seed)
    train_dataset = __make_dataset(train_df)
    test_df = newdf.sample(frac=0.2, random_state=random_seed)
    test_dataset = __make_dataset(test_df)
    return train_dataset, test_dataset

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    train, test = parse_data()


