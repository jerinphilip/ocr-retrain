                                                                        
from collections import Counter                                         
import re                                                               
from tokenizer import tokenize                                          
import sys                                                              
                                                                        
with open(sys.argv[1]) as ifp:                                          
    g = set()                                                           
    with open("output.log", "w") as ofp:                                
        count = 0                                                       
        counter = Counter()                                             
        for line in ifp:                                                
            count = count + 1                                           
            tokens = tokenize(line.strip())                             
            for token in tokens:                                        
                for c in token:                                         
                    g.add(c)                                            

    print(''.join(list(g)))
