import torch.nn.functional as F
from dataset import Dataset
from torch.utils.data import DataLoader
from arguments import get_args
from training import Trainer

def main():
    args = get_args()

    path = args['path']
    min_len = args['min_len']
    max_len = args['max_len']
    n_workers = args['worker']
    voc_n_keep = args['voc_size']
    batch_size = args['batch']
    shuffle = args['shuffle']
    embed_dim = args['embed']
    d_model = args['d_model']
    n_layers = args['layers']
    heads = args['heads']
    d_ff = args['dff']
    epochs = args['epoch']
    save_dir = args['save']
    loss_func = F.cross_entropy

    # Retrieves the dataset, cleans, processes and creates tensors from it
    training_set = Dataset(path, min_len, max_len, n_workers, voc_n_keep)

    vocab_size = training_set.vocab.num_words
    target_pad = training_set.vocab.PAD_token 
    
    # Pytorchs batch generator
    training_iter = DataLoader(training_set, batch_size, shuffle, num_workers=n_workers)

    # Creates model
    trainer = Trainer(vocab_size, embed_dim, d_model, n_layers, heads, d_ff, max_len)

    # Train model
    trainer.train(train_iter, loss_func, epochs, target_pad, save_dir)


if __name__ == '__main__': 
    main()
    