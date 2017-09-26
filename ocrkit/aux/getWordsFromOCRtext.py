import os
import sys
import re
sys.path.insert(0, '/home/deepayan/github/ocr-retrain/src/aux/') #specify the path where tokenizer file is present

from tokenizer import tokenize

def get_unicodes(path):
    out_lineno = []
    out_label=[]
    mydict = {}
    pattern = re.compile("output label string",re.IGNORECASE)

    with open(path) as in_file:
        for linenum, line in enumerate(in_file):
            mydict[linenum] = line.translate(None,'\t\n')

            if pattern.search(line.translate(None,'\t\n')) != None:

                out_lineno.append((linenum))

    for i in range(len(out_lineno)):

        out_label.append(mydict[(out_lineno[i]+1)])

    return(out_label)


def convert_unicodesTo_text(path):
    if os.path.exists('output.txt'):
        os.remove('output.txt')
    lines = get_unicodes(path)

    with open('output.txt','a') as out_file:
        for line in lines:
            each_line = line.split(' ')
            unicodes = '\u'+'\u'.join(each_line)+'0020'  #converts the output labels
                                                        # into unicode string

            try:
                unicodes_decoded = (unicodes.decode('unicode-escape'))
                out_file.write(unicodes_decoded.encode('utf-8')+'\n') #encodes the unicode into utf-8 and writes
                                                                       # into a text file
            except Exception as e:
                print e

convert_unicodesTo_text('features.out')

def get_words():
    words=[]
    if os.path.exists('words.txt'):
        os.remove('words.txt')
    with open('output.txt','r')as read_file:
        text = read_file.read()
        tokens = tokenize(text)

        for token in tokens:

            try:
                encoded = token.encode('utf-8')
                if encoded:
                    words.append(encoded)
            except Exception as e:
                print e



    thefile = open('words.txt' , 'w')          #extracts words from the text output file and writes
                                               # into a word.txt file
    thelist = list(set(words))
    for item in thelist:
        item = item.translate(None, '\n\t')
        thefile.write("%s\n" % item)

get_words()