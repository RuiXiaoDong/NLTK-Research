'''
Created on Jun 22, 2015

@author: dongx
'''
import nltk 
from nltk.book import *

print("-----raw content-----")
print(text1)

print("-----common context-----")
print(text2.common_contexts(["monstrous", "very"]))

print("-----similar-----")
print(text2.similar("very"))
#text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

fdist1 = FreqDist(text1)
fdist1.plot(50, cumulative=True)

print("-----collocations-----")
print(text8.collocations())