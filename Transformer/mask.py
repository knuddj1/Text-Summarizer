import torch
import numpy as np
from torch.autograd import Variable

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, pad):
    
    src_mask = (src != pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask    
    else:
        trg_mask = None
    return src_mask, trg_mask