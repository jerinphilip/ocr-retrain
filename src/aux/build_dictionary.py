#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
from xml.dom import minidom
import os
import sys

sys.path.insert(0, '/home/deepayan/github/ocr-retrain/src/aux/') #specify the path where tokenizer file is present

from tokenizer import tokenize

# the below code iterates over each book present in specified path and writes all the unique words in a text file.
# the text file can later be loaded as a list

path = sys.argv[1]        #path containg the books
language = sys.argv[2]    #specify the language
def language_dict():
    words=[]
    book_list = [x[0] for x in os.walk(path)]
    for each_dir in book_list[1:]:
        full_path = os.path.join(each_dir, 'text.xml')
        if os.path.exists(full_path):
            xmldoc = minidom.parse(full_path)
            rows = xmldoc.getElementsByTagName('row')
            for row in rows[:]:
                text = row.getElementsByTagName('field')[3].firstChild.data
                tokens = tokenize(text)


                for token in tokens:
                    encoded = token.encode('utf-8')
                    if encoded :
                        words.append(encoded)

    thefile = open('%s.txt'%language,'w')
    thelist = list(set(words))
    for item in thelist:
        item = item.translate(None,'\n\t')
        thefile.write("%s\n" % item)


language_dict()
