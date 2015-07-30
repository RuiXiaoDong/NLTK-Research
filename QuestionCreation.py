'''
Created on Jul 8, 2015

@author: dongx
'''
import nltk
from nltk.corpus import wordnet as wn
from nltk.tree import *
#Read sample text
with open('sample.txt', 'r') as f:
    sample = f.read()

global_entity_names = []
questions = []
sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

def ExtractWords( sentence, label):
    words = []
    for child in sentence:
        if (child[1] == label):
            words.extend([child[0]])
    return words

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

print("--------------Fill-in-the-Blank Question-------------")
for idx,tree in enumerate(chunked_sentences):
    entity_names = extract_entity_names(tree)
    sentence = sentences[idx]
    if(len(entity_names) > 0):
        for entity_name in entity_names:
            sentence = sentence.replace(entity_name, "____")
        questions.extend(sentence)
        print(sentence)
        print(entity_names)
        print()
    global_entity_names.extend(entity_names)
global_entity_names = list(set(global_entity_names))

print("--------------Entities-------------")
print(global_entity_names)
print()

print("--------------Multiple Choice Question-------------")
for entity_name in global_entity_names:
    for wn_element in wn.synsets(entity_name):
        member_holonyms = wn_element.member_holonyms()
        if(len(member_holonyms) > 0):
            for member_holonym in member_holonyms:
                for lemma in member_holonym.lemmas():
                    print("premise:",entity_name,"is",lemma.name().replace("_"," "))
        
'''
print("--------------Multiple Choice Question-------------")
for idx,tree in enumerate(chunked_sentences):
    entity_names = extract_entity_names(tree)
    if(len(entity_names) > 0):
        print(entity_names)
        for entity_name in entity_names:
            print("original entity:",entity_name)
            for wn_element in wn.synsets(entity_name):
                print("synonyms:",wn_element.hypernyms())
'''
'''
for idx,sent in enumerate(tagged_sentences):
    words = ExtractWords(sent, 'NN')
    if(len(words) > 0):
        for word in words:
            print("original words:",word)
            for wn_element in wn.synsets(word):
                print("synonyms:",wn_element.hypernyms())
'''
#Print unique entity names
#print(global_entity_names)