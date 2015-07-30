'''
Created on Jun 23, 2015

@author: dongx
'''
import nltk
from nltk.corpus import brown
hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
hobbies_learned.findall(r"<\w*> <is> <\w*>")