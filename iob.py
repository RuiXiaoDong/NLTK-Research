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

cp = nltk.RegexpParser("")
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
print("--------no pattern---------------------")
print(cp.evaluate(test_sents))

grammar = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar)
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
print("--------simple pattern---------------------")
print(cp.evaluate(test_sents))

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
unigram_chunker = UnigramChunker(train_sents)
print("--------unigram chunker---------------------")
print(unigram_chunker.evaluate(test_sents))
print(unigram_chunker.tagger.tag(treebank.sents()[0]))
