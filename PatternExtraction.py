'''
Created on Jun 19, 2015

@author: dongx
'''
import nltk
import nltk.corpus, nltk.tag, itertools
from nltk.tree import *
from nltk import Tree
from nltk import word_tokenize, sent_tokenize, Text,pos_tag
from nltk.tag import brill, brill_trainer, DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger

sample= """I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation.
Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous daybreak to end the long night of their captivity.
"""

#brown corpus trainning data
brown_review_sents = nltk.corpus.brown.tagged_sents(categories=['reviews'])
brown_lore_sents = nltk.corpus.brown.tagged_sents(categories=['lore'])
brown_romance_sents = nltk.corpus.brown.tagged_sents(categories=['romance'])
brown_train = list(itertools.chain(brown_review_sents[:1000], brown_lore_sents[:1000], brown_romance_sents[:1000]))
brown_test = list(itertools.chain(brown_review_sents[1000:2000], brown_lore_sents[1000:2000], brown_romance_sents[1000:2000]))
#conll corpus trainning data
conll_sents = nltk.corpus.conll2000.tagged_sents()
conll_train = list(conll_sents[:4000])
conll_test = list(conll_sents[4000:8000])
#treebank corpus trainning data
treebank_sents = nltk.corpus.treebank.tagged_sents()
treebank_train = list(treebank_sents[:1500])
treebank_test = list(treebank_sents[1500:3000])

"""
             "Barack Obama is the 44th and current president of the United States, and the first African American to serve as U.S. president."
             "Why would you go to school when you could work and earn money?",
             "the little yellow dogs barked at the cat.",
             "Juice is called lassi in Indian.",
             "London is the capital of England.",
             "Numbness on one side of your face, trouble speaking and dizziness are symptoms of a stroke.",
             "Earth is close to moon.",
             "Man in English means in Chinese.",
             "Taiwan belongs to China.",
             "Sneezing is a symptom of flu",
"""
sentences = ["Taiwan belongs to China."]

noun_phrase = 'NP'
verb_phrase = "VP"

# define a tag pattern of an NP chunk
grammar = '''
          P: {<IN>|<TO>}
          JP: {(<JJ.*><CC>)*<JJ.*>+}
          NP: {(<DT>|<PRP\$>)?<CD>?<JP>?<NN.*>+}
          NPS: {<NP>(<P><NP>)*}
          VP:{(<V.*><V.*>|<V.*>)<RB.*>?<P>?<NPS>?}
          '''

#Extract phrases according to the desire lable
def extractPhrases( myTree, myLabel):
    myPhrases = []
    if (myTree.label() == myLabel):
        myPhrases.append( myTree.copy(True) )
    for child in myTree:
        if (type(child) is Tree):
            list_of_phrases = extractPhrases(child, myLabel)
            if (len(list_of_phrases) > 0):
                myPhrases.extend(list_of_phrases)
    return myPhrases

#Convert tree into string format
def tree2Strng(t):
    leaves = [word for word, tag in t.leaves()]
    t_str = ' '.join(word for word, tag in t.leaves())
    return t_str

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
"""
default_tagger = DefaultTagger('NN')
initial_tagger = backoff_tagger(brown_train, [UnigramTagger, BigramTagger, TrigramTagger], backoff=default_tagger)
brill_tagger = train_brill_tagger(initial_tagger, brown_train)
"""
#Chunk target text into list of sentences
def chunkIntoWords( text ):
    words = word_tokenize(text)
    print words
    return words

#Chunk target text into list of words 
def chunkIntoSentences( text ):
    sentences = sent_tokenize(text)
    return sentences

def chunkIntoPhrases( text ):
    sentences = chunkIntoSentences( text )
    phrases = []
    for sentence in sentences:
        words = chunkIntoWords(sentence)
        tagged_words = nltk.pos_tag(words)
        cp = nltk.RegexpParser(grammar)
        tagged_phrases = cp.parse(tagged_words)
        list_np = extractPhrases(tagged_phrases, 'NP')
        list_jp = extractPhrases(tagged_phrases, 'JP')
        list_jj = extractPhrases(tagged_phrases, 'JJ')
        list_v = extractPhrases(tagged_phrases, 'V')
        for np in list_np:
            phrases.append(tree2Strng(np))
            
        for jp in list_jp:
            phrases.append(tree2Strng(jp))
            
        for jj in list_jj:
            phrases.append(tree2Strng(jj))
        
        for v in list_v:
            phrases.append(tree2Strng(v))
    print phrases
    return phrases

chunkIntoPhrases(sample)