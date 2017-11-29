from doctools.parser import read_book
from .opts import base_opts
from argparse import ArgumentParser
import json
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    config = json.load(open(args.config))

    book_name = config["books"][args.book]
    book_path = os.path.join(config["dir"], book_name)

    pagewise = read_book(book_path=book_path, unit='word')
    print(book_path)
    for imgs, truths in pagewise:
        m, n = len(imgs), len(truths)
        print(m, n, m==n)


