'''
Created on Jul 8, 2015

@author: dongx
'''
import nltk
import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def convertChunkMode( mode ):
    try: 
        return int(mode)
    except ValueError:
        return 1

def chunkIntoWords( text ):
    words = word_tokenize(text)
    return words

def chunkIntoSentences( text ):
    sentences = sent_tokenize(text)
    return sentences

content = """The more things change... Yes, I'm inclined to agree, especially with regards to the historical relationship between stock prices and bond yields. The two have generally traded together, rising during periods of economic growth and falling during periods of contraction. Consider the period from 1998 through 2010, during which the U.S. economy experienced two expansions as well as two recessions: Then central banks came to the rescue. Fed Chairman Ben Bernanke led from Washington with the help of the bank's current $3.6T balance sheet. He's accompanied by Mario Draghi at the European Central Bank and an equally forthright Shinzo Abe in Japan. Their coordinated monetary expansion has provided all the sugar needed for an equities moonshot, while they vowed to hold global borrowing costs at record lows."""
result_str = ""

print "Just word:"
tokens = chunkIntoWords(content)
print tokens

print "Just Sentence:"
tokens = chunkIntoSentences(content)
print tokens

print "Sentence then word:"
sentences = chunkIntoSentences(content)
for sentence in sentences:
    tokens = word_tokenize(sentence)
    print tokens
    