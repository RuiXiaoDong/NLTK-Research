'''
Created on Jul 20, 2015

@author: dongx
'''
import nltk
from nltk.corpus.reader import ConllChunkCorpusReader
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tree import Tree
from nltk.corpus import treebank
from nltk.corpus import conll2000

iob = tree2conlltags(Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])]))
tree = conlltags2tree([('the', 'DT', 'B-NP'), ('book', 'NN', 'I-NP')])

print("--------convertion between iob and tree---------------------")
print(iob)
print(tree)
