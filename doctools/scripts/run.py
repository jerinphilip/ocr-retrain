from doctools.simulate import Simulator
from doctools.ocr import GravesOCR
from doctools.postproc import Dictionary

from argparse import ArgumentParser
import os
import json

if __name__ == '__main__':
    parser = ArgumentParser(description="Run the simulation")
    parser.add_argument("-c", "--config", type=str, required=True, help="/path/to/config")
    parser.add_argument("-l", "--lang", type=str, required=True, help="language to run for")
    parser.add_argument("-b", "--book", type=int, required=True, help="id of the book to run simulation on")
    parser.add_argument("-o", "--output", type=str, required=True, help="output directory path")
    parser.add_argument("-n", "--batches", type=int, required=True, help="number of batches to run")

    args = parser.parse_args()

    config_file = open(args.config)
    config = json.load(config_file)

    # Initialize OCR, Error Module and Book Locs
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    paths = []
    for book in config["books"]:
        path = "%s%s"%(config["dir"], book)
        paths.append(path)

    book_name = config["books"][args.book]


    # Create the simulation
    simulation = Simulator(ocr=ocr, 
            postproc=error, 
            books=paths, 
            batches=args.batches)

    simulation.leave_one_out(args.book)
    simulation.recognize()
    stats = simulation.postprocess()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    langdir = os.path.join(args.output, args.lang)
    if not os.path.exists(langdir):
        os.mkdir(langdir)

    output_filename = "%s.json"%(book_name)
    output_fpath = os.path.join(langdir, output_filename)
    with open(output_fpath, "w+") as ofp:
        json.dump(stats, ofp, indent=4)

