from transformers import *
import sys
from transformers.modeling_openai import OpenAIGPTLMHeadAgenModel
from torch.nn import CrossEntropyLoss
from utils_g import *
from torch import utils

noise_frac = 0.4

def train(path='openai-gpt', mind=0):
    max_epochs = 4
    train_ds = process_in_g(TRAIN_G)
    savedir = './modelg/savedmodels'
    model = OpenAIGPTLMHeadAgenModel.from_pretrained(path) #model not on cuda
    if path == 'openai-gpt':
        model.resize_token_embeddings(tokenizer_g.vocab_size + num_added_token_g)
    training_generator = utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param, lr=1e-5)
    model.to(device_g)
    ini = 0
    criteria = CrossEntropyLoss()
    train_losses = []
    # Loop over epochs
    for epoch in range(mind, max_epochs):
        # Training
        savepath = savedir + str(epoch + 1)
        model.train()
        losssum = 0.0
        count = 0
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for local_batch, local_labels in enumerate(training_generator):
            # Transfer to GPUpri
            x = local_labels[0].to(device_g)
            e = local_labels[1].to(device_g)
            label = local_labels[2].to(device_g)
            outputs = model(x, e=e, labels=label)
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
        #print(get_gpu_memory_map())
        model.save_pretrained(savepath)
        train_losses.append(avg)

    loss_df = pd.DataFrame()
    loss_df["train_loss"] = train_losses
    loss_df.to_csv('loss_'+data+'.csv')

if __name__ == '__main__':
    train()
