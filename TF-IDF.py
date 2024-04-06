import pandas as pd
messages=pd.read_csv('sms',sep='\t',names=[u'label','message'])

print(messages)

import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordlemma=WordNetLemmatizer()

corpus=[]
for i in range(0,len(messages)):
    review=re.sub('a-zA-Z',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[wordlemma.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

print()
print("Corpus:")

#Creating Tf-Idf and ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_features=100)
X=tfidf.fit_transform(corpus).toarray()

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=10000,formatter=dict(float=lambda x:"%.3g"%x))
print(X)

#N-Gram
tfidf2=TfidfVectorizer(max_features=100,ngram_range=(2,2))
X2=tfidf2.fit_transform(corpus).toarray()
print(tfidf2.vocabulary_)
print('---------------------------')
print(X2)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, messages['label'], test_size=0.2,random_state=42)

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,classification_report
y_pred=clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))   


new_message = "Free Visa card for you. Apply now!"
new_corpus=[wordlemma.lemmatize(word.lower()) for word in re.sub('[a-zA-Z]',' ',new_message).split() if word.lower() not in stopwords.words('english')]
new_message=' '.join(new_corpus)
new_X=tfidf.transform([new_message]).toarray()
prediction=clf.predict(new_X)
print(f"\n New Message:'{new_message}' is classified as:{'spam' if prediction[0]=='spam' else 'not spam'}") 