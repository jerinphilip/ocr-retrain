from marisa_trie import Trie
import os
from Levenshtein import distance

class Dictionary:
    def __init__(self, *args, **kwargs):
        self.trie = Trie()
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        if 'lang' in kwargs:
            lang = kwargs['lang']
            savefile = path + lang + '.save'
            keysfile = path + lang + '.words'
            if os.path.exists(savefile):
                self.trie.load(savefile)
            elif os.path.exists(keysfile):
                with open(keysfile) as fp:
                    keys = fp.read().splitlines()
                    self.trie = Trie(keys)
            else:
                raise FileNotFoundError(path)
            self.preprocess(lang)
        else:
            raise KeyError("lang not set.")


    def preprocess(self,lang):
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        fname = path + lang +'.alphabet'
        if os.path.exists(fname):
            self.alphabet = open(fname).read()
        else:
            raise FileNotFoundError(path+fname)

    def error(self, word):
        return (1-int(word in self.trie))

    def old_suggest(self, word):
        rule = lambda x: distance(x, word) <= 3
        suggestions = list(filter(rule, self.trie))
        suggestions = sorted(suggestions, key=lambda x: distance(x, word))
        n = min(5, len(suggestions))
        return suggestions[:n]

    def suggest(self, word):
        candidates = list(self.edits1(word) or self.edits2(word))
        in_dictionary = list(filter(lambda x: x in self.trie, candidates))
        suggestions = sorted(in_dictionary, key=lambda x: distance(x, word))
        n = min(5, len(suggestions))
        return suggestions[:n]

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = self.alphabet 
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

