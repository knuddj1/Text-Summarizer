import torch
import torch.nn as nn
import time
import os
from Transformer.Model import Transformer
from Transformer.mask import create_masks

class Trainer:
    def __init__(self, vocab_size, embed_dim,
                 d_model, n_layers, heads, d_ff, max_seq_len):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.model, self.optim = self._setup_model()


    def _setup_model(self):
        print("Building model. .")
        model = Transformer( self.vocab_size, self.vocab_size, self.embed_dim,
                            self.d_model, self.n_layers, self.heads, 
                            self.d_ff, self.max_seq_len)
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("Model successfully built")
        print("Total parameters: ", sum([p.nelement() for p in model.parameters()]))

        optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        return model, optim


    def train(self, train_iter, loss_func, epochs, target_pad,
            save_dir, vocab, save_every=10, print_every=5000):

        print("Training started")    
    
        self.model.cuda()
        self.model.train()
        start = time.time()
        temp = start
        total_loss = 0
        
        for epoch in range(epochs):
        
            for i, batch in enumerate(train_iter):
                self.model.zero_grad()

                src = batch.transpose(0,1).cuda()
                
                trg_input = src[:, :-1]
                
                ys = src[:, 1:].contiguous().view(-1)
                
                src_mask, trg_mask = create_masks(src, trg_input, target_pad)
                
                preds = self.model(src, trg_input, src_mask, trg_mask)
                
                self.optim.zero_grad()
                
                loss = loss_func(preds.view(-1, preds.size(-1)), ys, ignore_index=target_pad)
                loss.backward()
                self.optim.step()
                
                total_loss += loss.item()
                if (i + 1) % print_every == 0:
                    loss_avg = total_loss / print_every
                    print("time = %dm, epoch %d, iter = %d, loss = %.3f, \
                    %ds per %d iters" % ((time.time() - start) // 60,
                    epoch + 1, i + 1, loss_avg, time.time() - temp,
                    print_every))
                    total_loss = 0
                    temp = time.time()

            if (epoch + 1) % save_every == 0:
                self.save(save_dir, vocab, epoch + 1)

            
    

    def save(self, save_dir, vocab, epoch):
        save_path = 'model'
        if save_dir:
            if os.path.exists(save_dir):
                save_path = os.path.join(save_dir, save_path)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = os.path.join(save_path, 'checkpoint {} epochs'.format(epoch))

        to_save = {
            'encoder': self.model.encoder.state_dict(),
            'decoder': self.model.decoder.state_dict(),
            'out': self.model.out.state_dict(),
            'vocab': vocab.__dict__,
            'optim': self.optim.state_dict(),
            'embed': self.embed_dim,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'heads': self.heads,
            'dff': self.d_ff,
            'max_len': self.max_seq_len
            }
        

        torch.save(to_save, save_path)
        

            

