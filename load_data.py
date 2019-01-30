import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import zipfile
import nltk
import os
import multiprocessing
import time
from gensim.parsing.preprocessing import *
from itertools import chain

CUSTOM_FILTERS = [
          lambda x: x.lower(),  # To lowercase
          lambda text: re.sub(r'https?:\/\/.*\s', '', text, flags=re.MULTILINE), #To Strip away URLs
          strip_tags,  # Remove tags from s using RE_TAGS.
          strip_non_alphanum,  # Remove non-alphabetic characters from s using RE_NONALPHA.
          strip_punctuation,  # Replace punctuation characters with spaces in s using RE_PUNCT.
          strip_numeric,  # Remove digits from s using RE_NUMERIC.
          strip_multiple_whitespaces,  # Remove repeating whitespace characters (spaces, tabs, line breaks) from s and turns tabs & line breaks into spaces using RE_WHITESPACE.
          remove_stopwords,  # Set of 339 stopwords from Stone, Denis, Kwantes (2010).
          lambda x: strip_short(x)
         ]


def split_sent(s, min_len, max_len):
    split = []
    while len(s) >= min_len:
        if len(s) < max_len:
            split.append(s)
            s = []
        else:
            split.append(s[:max_len])
            s = s[max_len:]
    return split


def process_sentence(s, min_len, max_len):
    prepared = []
    tokens = preprocess_string(s, CUSTOM_FILTERS)
    if len(tokens) >= min_len:
        if len(tokens) < max_len:
            prepared.append(tokens)
        else:
            for ss in split_sent(tokens, min_len, max_len):
                prepared.append(ss)
    return prepared


def filter_sentences(text, min_len, max_len):
    filtered = []
    sents = nltk.sent_tokenize(text)
    for s in sents:
        for ps in process_sentence(s, min_len, max_len):
            filtered.append(ps)
    return filtered


def read_zip_file(filepath):
    """Returns each file one at a time"""
    zfile = zipfile.ZipFile(filepath)
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        yield ''.join(str(l) for l in ifile.readlines())
        

# def process_file(f, min_len, max_len):
#     text = open(f, mode='r', encoding='utf-8-sig', errors='ignore').read()
#     return filter_sentences(text, min_len, max_len)

def retreive_data(dirpath, min_len, max_len, n_workers=None):
    start = time.time()
    pool = multiprocessing.Pool(n_workers)
    print('Extracting files from zip . .')
    iters = [(txt, min_len, max_len) for txt in read_zip_file(dirpath)]
    print('Finished. took {} seconds'.format(time.time() - start))
    print('Found {} files'.format(len(iters)))
    print('Processing files . .')
    result = pool.starmap(filter_sentences, iters)
    print('Finished. took {} seconds'.format(time.time() - start))
    return list(chain.from_iterable(result))
