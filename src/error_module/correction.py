from collections import Counter
import pandas
from aux.tokenizer import tokenize
from Levenshtein import distance

#with open('/home/avijit/corpus/english/eng_news_2015_10K/eng_news_2015_10K-sentences.txt','r') as in_file:
 #   text = in_file.read()
  #  dictionary = sorted(list(set(tokenize(text))))

class Correction:
    def __init__(self,*args,**kwargs):
        with open(kwargs['vocabulary'], encoding='utf-8') as fp:
                text = fp.read()
                dictionary = sorted(list(set(tokenize(text))))

    def error(self,word):
        return (1-int(word in dictionary))


    def bigrams(self,word):
        bigrm=[]
        for i in range(len(word)-1):
            bigrm.append(word[i]+self.word[i+1])
        #print(bigrm)
        return (bigrm)

    def contains(self, seq):
        indices=[]
        for wordnum,word in enumerate(dictionary):
            if seq in word:
                indices.append(wordnum)
        #print(indices)
        return(indices)

    def build_table(self,word):
        seqs = self.bigrams(word)
        mydict={}
        for each_seq in seqs:
            mydict[each_seq]=self.contains(each_seq)
        
        return(mydict)


    def suggest(self,word):
        all_words=[]
        suggestions=[]
        mydict=self.build_table(word)
        for key in mydict:
            all_words+=mydict[key]

        
        c=Counter(all_words)
        common_indices=c.most_common(10)
        for each_index in common_indices:
            suggestions.append(dictionary[each_index[0]])
        return(suggestions)

    def edit(self,word):
      probable_words=[]
      suggested_words=self.suggest(word)
      for each_word in suggested_words:
        edit_dist=distance(word,each_word)
        if edit_dist <= 3:
          probable_words.append(each_word)
      return(sorted(probable_words,key=len))




