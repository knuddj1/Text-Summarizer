from zipfile import ZipFile
import numpy as np
import torch


def get_glove(size, vocab, path):
    glove_emb_sizes = {50: 'glove.6B.50d.txt',
                       100: 'glove.6B.100d.txt',
                       200: 'glove.6B.200d.txt',
                       300: 'glove.6B.300d.txt'}

    glove = {}

    with ZipFile(path, 'r') as myzip:
        embeddings = myzip.read(glove_emb_sizes[size]).decode().split('\n')[:-1]
        for line in embeddings:
            word, emb = line.split(maxsplit=1)
            glove[word] = np.array(emb.split()).astype(np.float)

    pretrained_weights = np.zeros((vocab.num_words, size))

    for i in range(vocab.num_words):
        word = vocab.index2word[i]
        if word in glove:
            pretrained_weights[i] = glove[word]
        else:
            pretrained_weights[i] = np.random.normal(scale=0.6, size=(size, ))

    return torch.from_numpy(pretrained_weights)

