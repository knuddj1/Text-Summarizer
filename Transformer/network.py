import copy
import torch.nn as nn
from .layers import EncoderLayer, DecoderLayer
from .embed import Embedder
from .sublayers import Norm


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, d_model, n_layers,
                 heads, d_ff, max_seq_len, pretrained_emb, trainable, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, embed_dim, max_seq_len, pretrained_emb, trainable, dropout)
        self.layers = _get_clones(EncoderLayer(d_model, heads, d_ff, dropout), n_layers)
        self.norm = Norm(d_model)
        self.rescale = nn.Linear(embed_dim, d_model)

    def forward(self, src, mask):
        x = self.rescale(self.embed(src))
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, d_model, n_layers,
                 heads, d_ff, max_seq_len, pretrained_emb, trainable, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, embed_dim, max_seq_len, pretrained_emb, trainable, dropout)
        self.layers = _get_clones(DecoderLayer(d_model, heads, d_ff, dropout), n_layers)
        self.norm = Norm(d_model)
        self.rescale = nn.Linear(embed_dim, d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.rescale(self.embed(trg))
        for i in range(self.n_layers):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, d_model, n_layers,
                 heads, d_ff, max_seq_len, pretrained_emb, trainable, dropout):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, d_model,
                               n_layers, heads, d_ff, max_seq_len,
                               pretrained_emb, trainable, dropout)

        self.decoder = Decoder(vocab_size, embed_dim, d_model,
                               n_layers, heads, d_ff, max_seq_len,
                               pretrained_emb, trainable, dropout)

        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
