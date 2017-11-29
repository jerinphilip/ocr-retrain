from doctools.parser import read_book, text
import cv2
import codecs, sys, os
from .opts import base_opts
from argparse import ArgumentParser
import json
import numpy as np

def operate(bookPath, outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    gtFileName = os.path.join(outputDir, 'annotation.txt')
    gtFile = codecs.open(gtFileName,'w',encoding='utf8')
    pagewise = read_book(book_path=bookPath, unit='word')

    pageNo = 0
    for page in pagewise:
        images, truths = page
        pageNo += 1
        if len(images) == len (truths):
            wordNo=0
            for image in images: #if #images #truths mapping might go wrong
                wordNo += 1
                try:
                    image = 255*image
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    wordImageBaseName = str(pageNo) + '_' +  str(wordNo) + '.jpg'
                    wordImageName = os.path.join(outputDir, wordImageBaseName)
                    wordImageRelativePath = str(pageNo) + '_' +  str(wordNo) + '.jpg'
                    cv2.imwrite(wordImageName, image)
                    gtFile.write(wordImageRelativePath + ' ' + truths[wordNo-1] + '\n')
                except:
                    image = np.zeros((32, 32))
                    wordImageBaseName = str(pageNo) + '_' +  str(wordNo) + '.jpg'
                    wordImageName = os.path.join(outputDir, wordImageBaseName)
                    wordImageRelativePath = str(pageNo) + '_' +  str(wordNo) + '.jpg'
                    cv2.imwrite(wordImageName, image)
                    gtFile.write(wordImageRelativePath + ' ' + truths[wordNo-1] + '\n')
                    



if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()

    config_file = open(args.config)
    config = json.load(config_file)

    book_name = config["books"][args.book]
    path = os.path.join(config["dir"], book_name)

    output_dir = os.path.join(args.output, book_name)
    operate(path, output_dir)




