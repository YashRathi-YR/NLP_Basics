'''Tokenization:
    -it is the process of converting data(paragraphs) into tokens that maybe word or sentences
    -we use nltk library
    from nltk.tokeniser import  sent_tokenize,word_tokenize'''
corpus="""Hey This is Yash. 
How u Doin ! Blah
"""
print(corpus)
#sentence tokenize
from nltk.tokenize import sent_tokenize
documents=sent_tokenize(corpus)
print(documents)

#word tokenization
from nltk.tokenize import  word_tokenize

text=word_tokenize(corpus)
print(text)

#we can sepereate punctuations  from words using following code using wordpunct_tokenize
from nltk.tokenize import wordpunct_tokenize

print(wordpunct_tokenize(corpus))





