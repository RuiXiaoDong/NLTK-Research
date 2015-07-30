'''
Created on Jun 17, 2015

@author: dongx
'''
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext

article = """Girl: But you already have a Big Mac...
Hobo: Oh, this is all theatrical.
Girl: Hola amigo... 
Hobo: his is all theatrical.
我说: "U.S.A 你好啊".
U.S.A is the abbreviation of United States. To use statistical parameters such as mean and standard deviation reliably, you need to have a good estimator for them. The maximum likelihood estimates (MLEs) provide one such estimator. However, an MLE might be biased, which means that its expected value of the parameter might not equal the parameter being estimated."""

sentences = sent_tokenize(article)

for sentence in sentences:
    tokens = word_tokenize(sentence)
    #print(sentence)

text = webtext.raw('overheard.txt')

print(text)
sent_tokenizer = PunktSentenceTokenizer(text)
sents1 = sent_tokenizer.tokenize(text)
sents2 = sent_tokenize(text)

sents1_article = sent_tokenizer.tokenize(article)
sents2_article = sent_tokenize(article)

print(sents1[0])
print(sents2[0])
print()
print(sents1[677])
print(sents2[677])
print()
print(sents1[678])
print(sents2[678])
print()
print(sents1[679])
print(sents2[679])
print()
print(sents1_article)
print(sents2_article)