import sys

root = '/OCRData2/ocr/gpu/'
sys.path.insert(0, root)


# Test 1 - Parser
from parser import read_book
book = '/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0006/'
pagewise = read_book(book)

# Test 2 - DataLoader
from ocr.preproc import DataLoader 
loader = DataLoader(pagewise=pagewise)

print("Data split:",)
print(list(map(len, [loader.split.train, loader.split.val, loader.split.test])))


# Test-3 - Analyze dimensions
seq, target = loader.split.train[0]
#print(seq.shape)


from ocr.util import concat
# Test-4 - Test concat function
seq_targ_1, seq_targ_2 = loader.split.train[:2]
#print(concat(seq_targ_1, seq_targ_2))

from ocr.util import knot
train = knot(loader.split.train, max_width=8192)
print(train[0][1])
