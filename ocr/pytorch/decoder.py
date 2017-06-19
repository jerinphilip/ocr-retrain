import Levenshtein as Lev
import torch
from six.moves import xrange


class ArgMaxDecoder(Decoder):

    def __init__(self, labels):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])

    def convert_to_strings(self, sequences, sizes=None):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        for x in range(len(sequences)):
            string = self.convert_to_string(sequences[x])
            string = string[0:int(sizes.data[x])] if sizes is not None else string
            strings.append(string)
        return strings
    def process_strings(self, sequences, remove_repetitions=False):
        
        
        processed_strings = []
        for sequence in sequences:
            string = self.process_string(remove_repetitions, sequence).strip()
            processed_strings.append(string)
        return processed_strings

    def process_string(self, remove_repetitions, sequence):
        string = ''
        for i, char in enumerate(sequence):
            if char != self.int_to_char[self.blank_index]:
                if remove_repetitions and i != 0 and char == sequence[i - 1]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                else:
                    string = string + char
        return string

    def convert_to_string(self, sequence):
        return ''.join([self.int_to_char[i] for i in sequence])

    def decode(self, probs, sizes=None):
        
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        strings = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes)
        return self.process_strings(strings, remove_repetitions=True)
