'''
Created on Jun 22, 2015

@author: dongx
'''
import nltk
from nltk.corpus import brown
from nltk.tree import *

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

#Relations that we are looking for
grammar_NP = '''
             P: {<IN>|<TO>}      # Preposition
             V: {<V.*><VBN><P>?|<V.*><JJ.*><P>|<V.*>}          # Verb
             JP: {(<JJ.*><CC>)*<JJ.*>+}
             NP: {(<DT>|<PRP\$>)?<CD>?<JP>?<NN.*>+}
             NPS: {<NP><P><NP>(<P><NP>)*}
             Relation:{(<V><JJ.*>+<P>)|(<V><P>?)}
             '''
pattern_NP = nltk.RegexpParser(grammar_NP)

for sent in brown.sents():
    print("----------------------------")
    taggedTokens = nltk.pos_tag(sent)
    structuredTokens = pattern_NP.parse(taggedTokens)
    list_np = ExtractPhrases(structuredTokens, 'NP')
    list_v = ExtractPhrases(structuredTokens, 'V')
    if(len(list_np) == 2 and len(list_v) == 1):
        print(taggedTokens)
        print("relation:",list_np[0],list_v[0],list_np[1])
        for np in list_np:
            print(np)
        for v in list_v:
            print(v)
        structuredTokens.draw()