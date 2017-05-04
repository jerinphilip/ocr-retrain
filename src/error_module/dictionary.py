from marisa_trie import Trie
import os

class Dictionary:
    def __init__(self, *args, **kwargs):
        self.trie = Trie()
        if 'lang' in kwargs:
            lang = kwargs['lang']
            savefile = lang + '.save'
            keysfile = lang + '.words'
            if os.path.exists(savefile):
                self.trie.load(savefile)
            elif os.path.exists(keysfile):
                with open(keysfile) as fp:
                    keys = fp.read().splitlines()
                    self.trie = Trie(keys)
                    self.trie.save(savefile)

    def error(self, word):
        return (1-int(word in self.trie))

