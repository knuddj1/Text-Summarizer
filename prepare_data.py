import multiprocessing
import time
import torch
from nltk.util import pad_sequence

def encode_s(s, voc, oov):
    s_enc = []
    for w in s:
        w_enc = voc[w] if w in voc else oov
        s_enc.append(w_enc)
    return s_enc

def encode_sequences(seqs, voc, n_workers=None):
    print('Encoding sequences . .')
    start = time.time()
    pool = multiprocessing.Pool(n_workers)
    iters = [(s, voc.word2index, voc.UNK_token) for s in seqs]
    enc_seqs = pool.starmap(encode_s, iters)
    print('Encoding sequences finished. took {} seconds'.format(time.time() - start))
    return enc_seqs

def add_padding(seqs, max_len, pad_token):
    print('Padding sequences . .')
    start = time.time()
    for idx, seq in enumerate(seqs):
        seqs[idx] = (seqs[idx] + [0] * (max_len - len(seqs[idx])))
    print('Padding sequences finished. took {} seconds'.format(time.time() - start))
    return seqs

def prepare_inputs(seqs, voc, max_len, n_workers=None):
    enc_seqs = encode_sequences(seqs, voc, n_workers)
    padded_seqs = add_padding(enc_seqs, max_len, voc.PAD_token)
    print('Sequences to Tensors . .')
    start = time.time()
    seq_tensor = torch.tensor(padded_seqs)
    print('Converting sequences finished. took {} seconds'.format(time.time() - start))
    return torch.tensor(padded_seqs)