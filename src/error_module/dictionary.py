from marisa_trie import Trie
import os

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

