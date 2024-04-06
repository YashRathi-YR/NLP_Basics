import pandas as pd

messages = pd.read_csv('sms', sep='\t', names=["label", "message"])

print(messages)

import re
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
print(corpus)

#Create bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100,binary=True)

X=cv.fit_transform(corpus).toarray()

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000,formatter=dict(float=lambda x: "%.3f" %x))  
print(X)

# N-Gram
cv.vocabulary_

## Create the Bag OF Words model with ngram
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100,binary=True,ngram_range=(2,3))
X=cv.fit_transform(corpus).toarray()

cv.vocabulary_

print(X)
# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, messages['label'], test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
from sklearn.metrics import accuracy_score, classification_report
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Classify a new message
new_message = "Free Visa card for you. Apply now!"
new_corpus = [ps.stem(word.lower()) for word in re.sub('[^a-zA-Z]', ' ', new_message).split() if word.lower() not in stopwords.words('english')]
new_message = ' '.join(new_corpus)
new_X = cv.transform([new_message]).toarray()
prediction = clf.predict(new_X)
print(f"\nNew message: '{new_message}' is classified as: {'spam' if prediction[0] == 'spam' else 'not spam'}")