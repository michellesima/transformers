from utils_g import *
import torch
from transformers.modeling_openai import OpenAIGPTLMHeadAgenModel

mind = str(1)
savepath = './modelgp/savedmodels' + mind

modelg = OpenAIGPTLMHeadAgenModel.from_pretrained('openai-gpt')
modelg.resize_token_embeddings(tokenizer_g.vocab_size + num_added_token_g)
path = os.path.join(savepath, 'model.bin')
modelg.load_state_dict(torch.load(path))
modelg.to(device_g)
cat = modelg.cat_head


a = torch.FloatTensor([1,0, 0])
a = a.to(device_g)
a = cat(a)
print(a)
b = torch.FloatTensor(a.data)
print(b)
