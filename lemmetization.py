#Lemmetization is process of converting word to root words rather then root stem, without changing the meaning of word, kind of similar to stemming  but more accurate

words=['eating','eats','eaten','writing','writes','Partying','boosting']

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lemmatization=WordNetLemmatizer()

for  word in words:
    print(word,"-->",lemmatization.lemmatize(word)) 
#we can use pos(part of speech) to change the form of word, whatever we write in pos will be considered as form of that word like verb is v, noun is n,etc. The output of word will be provided based on form of that word.
'''[lemmetization.lemmetize(word,pos="___")]'''
     
