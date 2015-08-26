'''
Created on Jun 23, 2015

@author: dongx
'''
import nltk
raw = """DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government.  Supreme executive power derives from a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
print(tokens)

#There are three methods for word normalization
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
lemmas = nltk.WordNetLemmatizer()

print("PORTER:")
print([porter.stem(t) for t in tokens])

print("LANCASTER:")
print([lancaster.stem(t) for t in tokens])

print("WORDNET:")
print([lemmas.lemmatize(t) for t in tokens])

class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

    def concordance(self, word, width=50):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()

#search all the variation for the same word
print("Search word while considering its variation")
porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('lie')