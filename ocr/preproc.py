from random import shuffle
from math import floor, ceil
from collections import namedtuple

class DataLoader:
    def __init__(self, **kwargs):
        self.page_wise = kwargs['pagewise']
        self.sequences, self.targets = self.squashed(self.page_wise)
        self.count = len(self.sequences)

        default = {
                'split': { 'train': 0.8, 'test': 0.2},
                'random': False,
        }
        for key in default: 
            if key not in kwargs:
                kwargs[key] = default[key]
        self.split = self.split_data(**kwargs)

    def squashed(self, pagewise):
        super_imgs, super_truths = [], []
        for imgs, truths in pagewise:
            super_imgs.extend(imgs)
            super_truths.extend(truths)
        return (super_imgs, super_truths)
        
    def split_data(self, **kwargs):
        indices = list(range(self.count))
        if kwargs['random']:
            shuffle(indices) # Works inplace, as per doc.

        percent = kwargs['split']

        # Split indices
        total = self.count
        current = 0
        train_count = ceil(percent['train']*total)
        train_indices = indices[current:current+train_count]
        current += train_count

        test_indices = indices[current:]

        # Generate namedtuple based on indices. Post Process
        keys = ['train', 'test']
        SplitDT = namedtuple('split', keys)

        ls = []
        get = lambda i: (self.sequences[i], self.targets[i])
        ls.append(list(map(get, train_indices)))
        ls.append(list(map(get, test_indices)))

        return SplitDT(*ls)
