from transformers import *
from transformers.modeling_openai import OpenAIGPTLMHeadAgenModel
from utils_g import *
import torch

maxepoch = 10

def train():
    dataset = process_in_g(TRAIN_G)
    model = OpenAIGPTLMHeadAgenModel.from_pretrained('openai-gpt') #model not on cuda
    model.resize_token_embeddings(tokenizer_g.vocab_size + num_added_token_g)
    training_generator = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=False)
    param = model.parameters()
    optimizer = AdamW(param, lr=1e-5)
    model.to(device_g)
    train_losses = []
    savedir = './modelg/savedmodels'
    for epoch in range(maxepoch):
        savepath = savedir + str(epoch + 1)
        model.train()
        losssum = 0.0
        count = 0
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for local_batch, local_labels in enumerate(training_generator):
            x = torch.LongTensor(local_labels[0].data).to(device_g)
            e = torch.FloatTensor(local_labels[1].data).to(device_g)
            label = torch.LongTensor(local_labels[2].data).to(device_g)

            outputs = model(x, e=e, labels=label)
            loss, logits = outputs[:2]
            print(loss)
            losssum += loss
            count += 1
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        avg = losssum / count
        print(epoch + 1)
        print(avg)
        model.save_pretrained(savepath)
        train_losses.append(avg)
if __name__ == '__main__':
    train()
    #print(tok_li)
    #print(es)
