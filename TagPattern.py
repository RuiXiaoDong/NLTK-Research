'''
Created on Jun 19, 2015

@author: dongx
'''
import nltk
from nltk.tree import *
from nltk import word_tokenize,Text,pos_tag

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
#TARGET: {<V><NP>?<JJ.*>*<P>?}
for sentence in sentences:    
    words = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(words)
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tags)
    print(result)
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
    result.draw()
