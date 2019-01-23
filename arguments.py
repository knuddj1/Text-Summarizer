import argparse
import json
from multiprocessing import cpu_count


class ConfigAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            with open(values, 'r') as f:
                config = json.load(f)
        except FileNotFoundError as error:
            print(error)
            print('Not a valid config file')
            print('Make sure the path to the file is correct and it is in format .json')
            exit(0)

        
        setattr(namespace, self.dest, config)

def get_args():
    parser = argparse.ArgumentParser(description='Parameters for Sentence Embedder Model')
    parser.add_argument('-config', action=ConfigAction)
    parser.add_argument('-path', default=None)
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

    args = vars(parser.parse_args())

    if args['config']:
        return args['config']
    else:
        if args['path']:
            return args
        else:
            print('Dataset path must be supplied')
            exit(0)