'''
Created on Jun 23, 2015

@author: dongx
'''
import nltk
raw = """DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government.  Supreme executive power derives from a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
print(tokens)

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()

print("PORTER:")
stems = [porter.stem(t) for t in tokens]
print(stems)

print("LANCASTER:")
stems = [lancaster.stem(t) for t in tokens]
print(stems)

print("WORDNET:")
lemmas = nltk.WordNetLemmatizer()
print([lemmas.lemmatize(t) for t in tokens])

class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

    def concordance(self, word, width=20):
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

porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('lie')