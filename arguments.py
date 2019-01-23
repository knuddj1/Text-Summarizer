import argparse
from multiprocessing import cpu_count

def get_args():
    parser = argparse.ArgumentParser(description='Parameters for Sentence Embedder Model')
    parser.add_argument('-path', required=True)
    parser.add_argument('-min_len', default=10, type=int)
    parser.add_argument('-max_len', default=30, type=int)
    parser.add_argument('-worker', default=cpu_count(), type=int)
    parser.add_argument('-voc_size', default=30000, type=int)                       
    parser.add_argument('-batch', default=512, type=int)
    parser.add_argument('-shuffle', action='store_false')
    parser.add_argument('-embed', default=512, type=int)
    parser.add_argument('-d_model', default=512, type=int)
    parser.add_argument('-heads', default=8, type=int)
    parser.add_argument('-layers', default=6, type=int)
    parser.add_argument('-dff', default=2048, type=int)
    parser.add_argument('-epoch', default=10, type=int)
    parser.add_argument('-save', default=None)          
    return vars(parser.parse_args())