#Stemming reduces words to their base form, but it may not always accurately shrink words to their smallest size due to its simplifying nature.

words=['eating','eats','eaten','writing','writes','Partying','boosting']

#porter Stemmer

from nltk.stem import PorterStemmer
stemming=PorterStemmer()
for word in words:
    print(word+"--->"+stemming.stem(word)) 

#Regex Stemmer class
#in this technique  we use regular expressions to identify and remove form whereever the sign is added
from nltk.stem import RegexpStemmer
regex_stemmer = RegexpStemmer('ing$|s$|e$|able$',min=4)
for word in words:
    print(regex_stemmer.stem(word))

#Snoball Stemmmer
    #it is somewhat better then Porter Stemmer, it also changes words to lowercases
from nltk.stem import SnowballStemmer
snowballstemmer=SnowballStemmer("english")
for word in words:
    print(word,"-->",snowballstemmer.stem(word))
