from collections import Counter
import re
from tokenizer import tokenize
import sys

with open(sys.argv[1]) as ifp:
    with open("output.log", "w") as ofp:
        count = 0
        counter = Counter()
        for line in ifp:
            count = count + 1
            tokens = tokenize(line.strip())
            wrap = lambda x : "[%s]"%(x)
            #print('%d: %s'%(count, ''.join(map(wrap,tokens))))
            for token in tokens:
                counter[token] += 1
        print_freq = lambda x: "%s : %d"%(x[0], x[1])
        print_word = lambda x: x[0]
            
        ostrs = map(print_word, counter.most_common())
        ofp.write('\n'.join(ostrs))
