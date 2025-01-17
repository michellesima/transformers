from transformers import *
import sys
from torch.nn import CrossEntropyLoss
from utils import *
from torch import utils
from utils_dr import *

noise_frac = 0.4

def train(data, path='openai-gpt', mind=0):
    max_epochs = 4
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    if data == 'para':
        train_ds = parse_file_dr(TRAIN_DR, noi_frac=noise_frac, para=True)
        savedir = './modelp/savedmodels'
    elif data == 'mix':
        train_ds = parse_file_dr(TRAIN_DR, noi_frac=noise_frac, para=True)
        train_roc = parse_file_dr(ROC_TRAIN, noi_frac=noise_frac)
        train_ds.append(train_roc)
        print(train_ds.len())
        savedir = './modelmix/savedmodels'
    else:
        train_ds= parse_file_dr(ROC_TRAIN, noi_frac=noise_frac)
        savedir = './modelr/savedmodels'
    model = OpenAIGPTLMHeadModel.from_pretrained(path) #model not on cuda
    if path == 'openai-gpt':
        model.resize_token_embeddings(tokenizer_dr.vocab_size + num_added_token_dr)
    training_generator = utils.data.DataLoader(train_ds, batch_size=batchsize_dr, shuffle=True)
    param = model.parameters()
    optimizer = AdamW(param, lr=1e-5)
    model.to(device_dr)
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
            x, label = parse_model_inputs_dr(local_labels)
            outputs = model(x.to(device_dr), labels=label.to(device_dr))
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
    data = sys.argv[1]
    if len(sys.argv) == 2:
        train(data)
    else:
        model = './savedm/savedmodels' + sys.argv[2]
        train(data, model, int(sys.argv[2]))
