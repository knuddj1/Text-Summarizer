import torch
import torch.nn as nn
import time
import os
from Transformer.network import Transformer
from Transformer.mask import create_masks


class Trainer:
    def __init__(self, vocab_size, embed_dim, d_model, n_layers,
                 heads, d_ff, max_seq_len, pretrained, trainable, dropout):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.pretrained = pretrained
        self.trainable = trainable
        self.dropout = dropout
        self._setup_model()

    def _setup_model(self):
        print("Building model. .")
        self.model = Transformer(self.vocab_size, self.embed_dim,
                                 self.d_model, self.n_layers, self.heads,
                                 self.d_ff, self.max_seq_len, self.pretrained,
                                 self.trainable, self.dropout)
        
        for p in self.model.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        print("Model successfully built")
        total_params = sum([p.nelement() for p in self.model.parameters()])
        trainable_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print("Trainable parameters = {0}/{1} ".format(trainable_params, total_params))

        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def train(self, train_iter, loss_func, epochs, target_pad,
              save_dir, vocab, save_every=10, print_every=5000):

        print("Training started")    
    
        self.model.cuda()
        self.model.train()
        start = time.time()

        for epoch in range(epochs):
            total_loss = 0
            for i, batch in enumerate(train_iter):
                self.model.zero_grad()

                src = batch.transpose(0, 1).cuda()
                
                trg_input = src[:, :-1]
                
                ys = src[:, 1:].contiguous().view(-1)
                
                src_mask, trg_mask = create_masks(src, trg_input, target_pad)
                
                preds = self.model(src, trg_input, src_mask, trg_mask)
                
                self.optim.zero_grad()
                
                loss = loss_func(preds.view(-1, preds.size(-1)), ys, ignore_index=target_pad)
                loss.backward()
                self.optim.step()
                
                total_loss += loss.item()
                if (i + 1) % print_every == 0 or (i + 1) == len(train_iter):
                    loss_avg = total_loss / (i + 1)
                    print("time = %dm, epoch %d, iter = %d, loss = %.3f" % ((time.time() - start) // 60,
                                                                            epoch + 1, i + 1, loss_avg))

            if (epoch + 1) % save_every == 0:
                self.save(save_dir, vocab, epoch + 1, total_loss / len(train_iter))

    def save(self, save_dir, vocab, epoch, loss):
        save_path = 'model'
        if save_dir:
            if os.path.exists(save_dir):
                save_path = os.path.join(save_dir, save_path)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = os.path.join(save_path, 'checkpoint {} epochs'.format(epoch))

        to_save = {'loss': loss,
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
                   'dropout': self.dropout,
                   'max_len': self.max_seq_len}

        torch.save(to_save, save_path)
