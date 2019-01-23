import torch
import torch.nn as nn
from Model import Transformer
from mask import create_masks

embed_dim = 512
d_model = 512
heads = 8
N = 6
d_ff = 2048
max_seq_len = 30
src_vocab = 30000
trg_vocab = 30000

path = 'model.pth'

model = Transformer(src_vocab, trg_vocab, embed_dim, d_model, N, heads, d_ff, max_seq_len)

transformer.load(path)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# this code is very important! It initialises the parameters with a
# range of values that stops the signal fading or getting too big.
# See this blog for a mathematical explanation.
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

print("Total Parameters:", sum([p.nelement() for p in transformer.parameters()]))

# transformer.save(path)

def train_model(epochs, print_every=100):
    
    model.train()
    
    start = time.time()
    temp = start
    
    total_loss = 0
    
    for epoch in range(epochs):
       
        for i, batch in enumerate(train_iter):
            src = batch.transpose(0,1)
            
            trg_input = src[:, :-1]
            
            # the words we are trying to predict
            
            ys = src[:, 1:].contiguous().view(-1)
            
            # create function to make masks using mask code above
            
            src_mask, trg_mask = create_masks(src, trg_input)
            
            preds = model(src, trg_input, src_mask, trg_mask)
            
            optim.zero_grad()
            
            loss = nn.functional.cross_entropy(preds.view(-1, preds.size(-1)), results, ignore_index=target_pad)
            loss.backward()
            optim.step()
            
            total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,
                %ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, i + 1, loss_avg, time.time() - temp,
                print_every))
                total_loss = 0
                temp = time.time()