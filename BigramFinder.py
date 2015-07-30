'''
Created on Jul 14, 2015

@author: dongx
'''
from nltk.corpus import webtext
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

stopset = set(stopwords.words('english'))
words = [w.lower() for w in webtext.words('grail.txt')]
filter_stops = lambda w: len(w) < 3 or w in stopset
bcf = BigramCollocationFinder.from_words(words)
bcf.apply_word_filter(filter_stops)
print(bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4))
