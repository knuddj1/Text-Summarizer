from load_data import retreive_data
from vocabulary import generate_vocab
from prepare_data import prepare_inputs


def get_data(path, min_len, max_len, n_workers, voc_n_keep):
    sequences = retreive_data(path, min_len, max_len, n_workers)
    vocab = generate_vocab(sequences, voc_n_keep)
    return prepare_inputs(sequences, vocab, max_len, n_workers), vocab


class Dataset:
    def __init__(self, path, min_len, max_len, n_workers, voc_n_keep):
        self.samples, self.vocab = get_data(path, min_len, max_len, n_workers, voc_n_keep)
        
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        return self.samples[idx]
