'''
Created on Jun 19, 2015

@author: dongx
'''
import nltk
import nltk.corpus, nltk.tag, itertools
from nltk.tree import *
from nltk import word_tokenize,Text,pos_tag
from nltk.tag import brill, brill_trainer, DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger

def ExtractPhrases( myTree, myLabel):
    myPhrases = []
    if (myTree.label() == myLabel):
        myPhrases.append( myTree.copy(True) )
    for child in myTree:
        if (type(child) is Tree):
            list_of_phrases = ExtractPhrases(child, myLabel)
            if (len(list_of_phrases) > 0):
                myPhrases.extend(list_of_phrases)
    return myPhrases

brown_review_sents = nltk.corpus.brown.tagged_sents(categories=['reviews'])
brown_lore_sents = nltk.corpus.brown.tagged_sents(categories=['lore'])
brown_romance_sents = nltk.corpus.brown.tagged_sents(categories=['romance'])
 
brown_train = list(itertools.chain(brown_review_sents[:1000], brown_lore_sents[:1000], brown_romance_sents[:1000]))
brown_test = list(itertools.chain(brown_review_sents[1000:2000], brown_lore_sents[1000:2000], brown_romance_sents[1000:2000]))
 
conll_sents = nltk.corpus.conll2000.tagged_sents()
conll_train = list(conll_sents[:4000])
conll_test = list(conll_sents[4000:8000])
 
treebank_sents = nltk.corpus.treebank.tagged_sents()
treebank_train = list(treebank_sents[:1500])
treebank_test = list(treebank_sents[1500:3000])

sentences = ["Why would you go to school when you could work and earn money?",
             "the little yellow dogs barked at the cat.",
             "Juice is called lassi in Indian.",
             "London is the capital of England.",
             "Numbness on one side of your face, trouble speaking and dizziness are symptoms of a stroke.",
             "Earth is close to moon.",
             "Man in English means 男人 in Chinese.",
             "Taiwan belongs to China.",
             "Sneezing is a symptom of flu",
             "Barack Obama is the 44th and current president of the United States, and the first African American to serve as U.S. president."]

# define a tag pattern of an NP chunk
grammar = '''
             P: {<IN>|<TO>}      # Preposition
             V: {<V.*><VBN><P>?|<V.*><JJ.*><P>|<V.*>}          # Verb
             JP: {(<JJ.*><CC>)*<JJ.*>+}
             NP: {(<DT>|<PRP\$>)?<CD>?<JP>?<NN.*>+}
             NPS: {<NP><P><NP>(<P><NP>)*}
             VP:{(<V><JJ.*>+<P>)|(<V><P>?)}
          '''
          
def backoff_tagger(tagged_sents, tagger_classes, backoff=None):
    if not backoff:
        backoff = tagger_classes[0](tagged_sents)
        del tagger_classes[0]
 
    for cls in tagger_classes:
        tagger = cls(tagged_sents, backoff=backoff)
        backoff = tagger
 
    return backoff

word_patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*ould$', 'MD'),
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*ness$', 'NN'),
    (r'.*ment$', 'NN'),
    (r'.*ful$', 'JJ'),
    (r'.*ious$', 'JJ'),
    (r'.*ble$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*ive$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*est$', 'JJ'),
    (r'^a$', 'PREP'),
]

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

default_tagger = DefaultTagger('NN')
initial_tagger = backoff_tagger(brown_train, [UnigramTagger, BigramTagger, TrigramTagger], backoff=default_tagger)
brill_tagger = train_brill_tagger(initial_tagger, brown_train)

#TARGET: {<V><NP>?<JJ.*>*<P>?}
for sentence in sentences:    
    words = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(words)
    brill_tags = brill_tagger.tag(words)
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tags)
    brill_result = cp.parse(brill_tags)
    print(result)
    print(brill_result)
    '''
    list_np = ExtractPhrases(result, 'NP')
    list_jp = ExtractPhrases(result, 'JP')
    list_jj = ExtractPhrases(result, 'JJ')
    list_v = ExtractPhrases(result, 'V')
    
    for np in list_np:
        print(np)
        
    for jp in list_jp:
        print(jp)
        
    for jj in list_jj:
        print(jj)
    '''
    result.draw()
    brill_result.draw()
