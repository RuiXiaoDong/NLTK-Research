'''
Created on Jul 22, 2015

@author: dongx
'''
import nltk
def rhyme(inp, level):
    entries = nltk.corpus.cmudict.entries()
    word_syllable_sets = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in word_syllable_sets:
        for (word_entry, syllable_entry) in entries:
            #search from the end of the array
            if syllable_entry[-level:] == syllable[-level:]:
                print(word_entry)
                rhymes += [word_entry]
                
    return set(rhymes)

def doTheyRhyme(word1, word2):
    #Check whether word1 or word2 is the other's substring
    #if word1.find(word2) == len(word1) - len(word2):
    #    return False
  #  if word2.find(word1) == len(word2) - len(word1): 
   #     return False
    return word1 in rhyme(word2, 1)
#Soundex is R500 and C500 in this case 
print(doTheyRhyme("desert", "desert"))