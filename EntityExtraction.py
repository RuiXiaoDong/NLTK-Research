'''
Created on Jul 6, 2015

@author: dongx
'''
import nltk 
#Read sample text
with open('sample.txt', 'r') as f:
    sample = f.read()

sample= """I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation. Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous daybreak to end the long night of their captivity."""
#Record all the extracted entities
global_entities = []
sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
ne_chunks = nltk.ne_chunk_sents(tagged_sentences, binary=True)

#Grab the contain from the NE chunk
def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

#Iterate through those NE chunks which contain entities
for idx,tree in enumerate(ne_chunks):
    entity_names = extract_entity_names(tree)
    global_entities.extend(entity_names)

#Print unique entity names
global_entities = list(set(global_entities))
print global_entities