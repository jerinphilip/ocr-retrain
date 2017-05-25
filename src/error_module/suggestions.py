from collections import Counter
import pandas
from aux.tokenizer import tokenize
from Levenshtein import distance

with open('/home/avijit/corpus/english/eng_news_2015_10K/eng_news_2015_10K-sentences.txt','r') as in_file:
    text = in_file.read()
    dictionary = sorted(list(set(tokenize(text))))

class Correction:
    def __init__(self,word):
        self.word = word
        

    def bigrams(self):
        bigrm=[]
        for i in range(len(self.word)-1):
            bigrm.append(self.word[i]+self.word[i+1])
        #print(bigrm)
        return (bigrm)

    def contains(self, seq):
        indices=[]
        for wordnum,word in enumerate(dictionary):
            if seq in word:
                indices.append(wordnum)
        #print(indices)
        return(indices)

    def build_table(self):
        seqs = self.bigrams()
        mydict={}
        for each_seq in seqs:
            mydict[each_seq]=self.contains(each_seq)
        
        return(mydict)


    def suggest(self):
        all_words=[]
        suggestions=[]
        mydict=self.build_table()
        for key in mydict:
            all_words+=mydict[key]

        
        c=Counter(all_words)
        common_indices=c.most_common(10)
        for each_index in common_indices:
            suggestions.append(dictionary[each_index[0]])
        return(suggestions)

    def edit(self):
      probable_words=[]
      suggested_words=self.suggest()
      for each_word in suggested_words:
        edit_dist=distance(self.word,each_word)
        if edit_dist <= 3:
          probable_words.append(each_word)
      return(sorted(probable_words,key=len))




sug = Correction('cuty')
print(sug.edit())