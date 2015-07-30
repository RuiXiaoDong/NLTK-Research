'''
Created on Jul 8, 2015

@author: dongx
'''
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

article = """Hola amigo. Estoy bien. 我说: "U.S.A 你好啊". 今天我们一起去吧. I can't parse it. U.S.A is the abbreviation of United States. To use statistical parameters such as mean and standard deviation reliably, you need to have a good estimator for them. The maximum likelihood estimates (MLEs) provide one such estimator. However, an MLE might be biased, which means that its expected value of the parameter might not equal the parameter being estimated."""
sentences = sent_tokenize(article)

for sentence in sentences:
    tokens = word_tokenize(sentence)
    print(tokens)