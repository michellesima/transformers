from utils_g import *
from utils_ivp import agen_vector
import torch
from transformers.modeling_openai import OpenAIGPTLMHeadAgenModel
import sys

if __name__ == '__main__':
    mind = sys.argv[1]
    savepath = './modelgp/savedmodels' + mind

    modelg = OpenAIGPTLMHeadAgenModel.from_pretrained('openai-gpt')
    modelg.resize_token_embeddings(tokenizer_g.vocab_size + num_added_token_g)
    path = os.path.join(savepath, 'model.bin')
    modelg.load_state_dict(torch.load(path))
    cat = modelg.cat_head

    agen_vectors = agen_vector(tokenizer_g, num_added_token_g, multi=False)
    catlist = ['pos', 'equal', 'neg']
    for i in range(len(catlist)):
        
        a = torch.FloatTensor([0,0, 0])
        a[i] = 1
        a = cat(a)
        print('For ', catlist[i])
        print('mean of vector', torch.mean(a))
        print('std of vector', a.std())
        agen_vector = agen_vectors[catlist[i]]
        verbs = agen_vector.nonzero()
        product = agen_vector * a
        # non zero entries are verbs in that category
        gz = product > 0
        pgz = product[gz]
        nz = product != 0
        vnz = a[nz]
        print('mean of verb in descat ', torch.mean(vnz))
        print('std of verb in descat ', vnz.std())
        gz_cat = pgz.size()[0] / verbs.size()[0]
        
        print('percent of ', catlist[i], ' verbs with logits greater than 0 is ', gz_cat)
