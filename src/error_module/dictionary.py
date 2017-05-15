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


    def error(self, word):
        return (1-int(word in self.trie))

    def suggest(self, word):
        rule = lambda x: distance(x, word) <= 3
        suggestions = list(filter(rule, self.trie))
        suggestions = sorted(suggestions, key=lambda x: distance(x, word))
        n = min(5, len(suggestions))
        return suggestions[:n]


