from operator import add
import re

def tokenize(iota):
    punctuations = ' ,."\'/%();:!-?'
    tokens = re.split('[%s]'%(punctuations), iota)
    return tokens
