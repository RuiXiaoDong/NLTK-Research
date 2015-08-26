'''
Created on Jul 2, 2015

@author: dongx
'''

import nltk
from nltk.corpus import brown, treebank
from nltk.tag import untag, tnt, DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger, brill, brill_trainer
from nltk.corpus.reader import ChunkedCorpusReader

#trainning data
sent = treebank.sents()[0]
brown_train_sents = brown.tagged_sents(categories='news')[1001:]
brown_test_sents = brown.tagged_sents(categories='news')[:1000]

#form multiple tagger in a tagging chain
def backoff_tagger(tagged_sents, tagger_classes, backoff=None):
    if not backoff:
        backoff = tagger_classes[0](tagged_sents)
        del tagger_classes[0]
 
    for cls in tagger_classes:
        tagger = cls(tagged_sents, backoff=backoff)
        backoff = tagger
 
    return backoff

def train_brill_tagger(initial_tagger, train_sents, **kwargs):
    templates = [
        brill.Template(brill.Pos([-1])), #a rule can be generated using the previous part-of-speech tag
        brill.Template(brill.Pos([1])),
        brill.Template(brill.Pos([-2])),
        brill.Template(brill.Pos([2])),
        brill.Template(brill.Pos([-2, -1])),
        brill.Template(brill.Pos([1, 2])),
        brill.Template(brill.Pos([-3, -2, -1])),
        brill.Template(brill.Pos([1, 2, 3])),
        brill.Template(brill.Pos([-1]), brill.Pos([1])),
        brill.Template(brill.Word([-1])),
        brill.Template(brill.Word([1])),
        brill.Template(brill.Word([-2])),
        brill.Template(brill.Word([2])),
        brill.Template(brill.Word([-2, -1])), #you can look at the combination of the previous two words to learn a transformation rule
        brill.Template(brill.Word([1, 2])),
        brill.Template(brill.Word([-3, -2, -1])),
        brill.Template(brill.Word([1, 2, 3])),
        brill.Template(brill.Word([-1]), brill.Word([1])),
    ]
    
    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates, deterministic=True)
    return trainer.train(train_sents, **kwargs)

defaultTagger = DefaultTagger('NN')
initialTagger = backoff_tagger(brown_train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=defaultTagger)
brillTagger = train_brill_tagger(initialTagger, brown_train_sents)

tnt_tagger = tnt.TnT(N=100)
tnt_tagger.train(brown_train_sents)

bigramTagger = BigramTagger(brown_train_sents)
trigramTagger = TrigramTagger(brown_train_sents)

print("------------Recommended Tagger------------")
print(nltk.pos_tag(sent))

print("------------Default Tagger------------")
print(defaultTagger.tag(sent))

print("------------Unigram Tagger Overrode------------")
unigramTagger = UnigramTagger(model={'Pierre': 'NN'})
print(unigramTagger.tag(sent))

print("------------Unigram Tagger Trained------------")
unigramTagger = UnigramTagger(brown_train_sents)
print(unigramTagger.tag(sent))

#cutoff: The number of instances of training data the tagger must see in order not to use the backoff tagger
print("------------Unigram Tagger Trained with cutoff=3------------")
unigramTagger = UnigramTagger(brown_train_sents, cutoff=3)
print(unigramTagger.tag(sent))

print("------------Bigram Tagger------------")
print(bigramTagger.tag(sent))

print("------------Trigram Tagger------------")
print(trigramTagger.tag(sent))

print("------------Brill Tagger------------")
print(brillTagger.tag(sent))

print("------------Accuracy: Unigram Tagger Trained------------")
unigramTagger = UnigramTagger(brown_train_sents)
print(unigramTagger.evaluate(brown_test_sents))

print("------------Accuracy: Unigram Tagger Trained with cutoff = 3------------")
unigramTagger = UnigramTagger(brown_train_sents, cutoff = 3)
print(unigramTagger.evaluate(brown_test_sents))

print("------------Accuracy: Bigram Tagger Trained------------")
print(bigramTagger.evaluate(brown_test_sents))

print("------------Accuracy: Trigram Tagger Trained------------")
print(trigramTagger.evaluate(brown_test_sents))

print("------------Accuracy: Unigram Tagger with backoff enabled. Backoff Chain: UnigramTagger -> DefaultTagger------------")
unigramTagger = UnigramTagger(brown_train_sents, backoff=defaultTagger)
print(unigramTagger.evaluate(brown_test_sents))

print("------------Accuracy: Tagger with backoff enabled. Backoff Chain: TrigramTagger -> BigramTagger -> UnigramTagger -> DefaultTagger------------")
print(initialTagger.evaluate(brown_test_sents))

print("------------Accuracy: Brill Tagger------------")
print(brillTagger.evaluate(brown_test_sents))
print(brillTagger.rules())

print("------------Accuracy: TnT Tagger------------")
print(tnt_tagger.evaluate(brown_test_sents))
