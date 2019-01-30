import time


class Vocab:

    PAD_token = 0  # Used for padding short sentences
    UNK_token = 1  # Used for words not in the vocabulary
    SOS_token = 2  # Start-of-sentence token
    EOS_token = 3  # End-of-sentence token

    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            Vocab.PAD_token: "<PAD>",
            Vocab.UNK_token: "<UNK>",
            Vocab.SOS_token: "<SOS>",
            Vocab.EOS_token: "<EOS>"
        }
        self.num_words = len(self.index2word)  # Count <SOS>, <EOS>, <PAD>, <UNK>

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, n_keep):
        if self.trimmed:
            return
        if n_keep > len(self):
            return

        self.trimmed = True

        keep_words = []

        sorted_word_freqs = sorted(self.word2count.items(), key=lambda x: x[1], reverse=True)

        for i in range(n_keep):
            keep_words.append(sorted_word_freqs[i][0])

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.__init_dictionaries()

        for word in keep_words:
            self.add_word(word)

    def __init_dictionaries(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            Vocab.PAD_token: "<PAD>",
            Vocab.UNK_token: "<UNK>",
            Vocab.SOS_token: "<SOS>",
            Vocab.EOS_token: "<EOS>"
        }
        self.num_words = len(self.index2word)  # Count default tokens

    def __len__(self):
        return self.num_words


def generate_vocab(sequences, n_keep):
    start = time.time()
    print('Creating Vocabulary . .')    
    voc = Vocab()
    for s in sequences:
        voc.add_sentence(s)
    voc.trim(n_keep)
    print('Finished. took {} seconds'.format(time.time() - start))
    return voc
