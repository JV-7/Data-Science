import nltk
# nltk.download('punkt') # this is one time installation
from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download('all')

text = ('NLTK is wonderful Library for NLP. This shows greater structure to the tools and text for various processing '
        'within')

sentence = sent_tokenize(text=text)
word = word_tokenize(text=text)

print(f''' First One
{sentence}
{word}
''')

from nltk import pos_tag
# nltk.download('averaged_perceptron_tagger')
word1 = word_tokenize(text)
tag = pos_tag(word1)

print(f'First tag \n{tag}')

''' Named Entity Recognition '''


from nltk.chunk import ne_chunk

# Tokenize the sentence
word2 = word_tokenize("Sachin is 1st player to Score 200 in Cricket world")

# Perform POS tagging
tag1 = pos_tag(word2)

# Perform Named Entity Recognition (NER)
named_entity = ne_chunk(tag1)
print('''\n Named Entity Recognition \n''')
print(tag1)
print(word2)
print(named_entity)


