import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, pretrained_emb, trainable):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        if pretrained_emb is not None:
            self.embed.load_state_dict({'weight': pretrained_emb})
        if not trainable:
            self.embed.weight.requires_grad = False
        
    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, emb_size, max_seq_len, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, emb_size)
        for pos in range(max_seq_len):
            for i in range(0, emb_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / emb_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / emb_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.emb_size)
        # 1add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe.transpose(0, 1)
        return self.dropout(x)


class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_size, max_seq_len, pretrained_emb, trainable, dropout):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, emb_size, pretrained_emb, trainable)
        self.position = PositionalEncoder(emb_size, max_seq_len, dropout)

    def forward(self, x):
        x = self.token(x)
        return self.position(x)
