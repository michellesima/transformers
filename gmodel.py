from transformers import *
from utils_g import *
import torch.nn as nn

class ModelGenWeight(nn.Module):
    def __init__(self):
        self.gpt = OpenAIGPTLMHeadModel.from_pretrained(path) #model not on cuda
        self.gpt.resize_token_embeddings(tokenizer_g.vocab_size + num_added_token_g)
        self.fc1 = 
