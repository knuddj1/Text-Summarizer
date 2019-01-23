import torch
import torch.nn as nn 
from .Layers import EncoderLayer, DecoderLayer
from .Embed import Embedder
from .Sublayers import Norm
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, d_model, N, heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, embed_dim, max_seq_len, dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, d_ff, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, d_model, N, heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, embed_dim, max_seq_len, dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, d_ff, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, embed_dim, d_model, N, heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, d_model, N, heads, d_ff, max_seq_len, dropout)
        self.decoder = Decoder(trg_vocab, embed_dim, d_model, N, heads, d_ff, max_seq_len, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output