import numpy as np

def read_features(filename):
    mat = np.load(filename)
    return mat

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-f', '--feature_file', required=True)

    args = parser.parse_args()
    print(read_features(args.feature_file))
