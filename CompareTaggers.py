'''
Created on Jul 2, 2015

@author: dongx
'''
import nltk
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.tag import untag
from nltk.tag import DefaultTagger
from nltk.tag import UnigramTagger
from nltk.corpus.reader import ChunkedCorpusReader


sent = treebank.sents()[0]
brown_train_sents = brown.tagged_sents(categories='news')[1000:]
brown_test_sents = brown.tagged_sents(categories='news')[:1000]

print("------------Recommended Tagger------------")
print(nltk.pos_tag(sent))
    
print("------------Default Tagger------------")
defaultTagger = DefaultTagger('NN')
print(defaultTagger.tag(sent))
    
print("------------Unigram Tagger Untrained------------")
unigramTagger = UnigramTagger(brown.tagged_sents(categories='news')[:1])
print(unigramTagger.tag(sent))

print("------------Unigram Tagger Overrode------------")
unigramTagger = UnigramTagger(model={'Pierre': 'NN'})
print(unigramTagger.tag(treebank.sents()[0]))

print("------------Unigram Tagger Trained------------")
#cutoff: The number of instances of training data the tagger must see in order not to use the backoff tagger
unigramTagger = UnigramTagger(brown_train_sents, cutoff=3)
print(unigramTagger.tag(treebank.sents()[0]))

print("------------Accuracy: Unigram Tagger Trained------------")
unigramTagger = UnigramTagger(brown_train_sents)
print(unigramTagger.evaluate(brown_test_sents))

print("------------Accuracy: Unigram Tagger Trained with cutoff = 3------------")
unigramTagger = UnigramTagger(brown_train_sents, cutoff = 3)
print(unigramTagger.evaluate(brown_test_sents))

print("------------Accuracy: Unigram Tagger Trained with Default Tagger as a backoff tagger------------")
unigramTagger = UnigramTagger(brown_train_sents, backoff=defaultTagger)
print(unigramTagger.evaluate(brown_test_sents))