from aux.functional import compose
from operator import add
import re

def tokenize(iota):
    lines = iota.split('\n')
    punctuations = ' ,."\'/%();:!-?'
    tokens = re.split('[%s]'%(punctuations), iota)
    return tokens

