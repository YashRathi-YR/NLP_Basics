
import pandas as pd
data=pd.read_csv('all_kindle_review .csv')
data.head()



df=data[['reviewText','rating']]
df.head()

df.shape

df.isnull().sum()

df['rating'].unique()

df['rating'].value_counts()
import re
# Instead of this:
# df['rating'] = df['rating'].apply(lambda x: 0 if x < 3 else 1)
df.loc[:, 'rating'] = df['rating'].apply(lambda x: 0 if x < 3 else 1)

# Instead of this:
# df['reviewText'] = df['reviewText'].str.lower()
df.loc[:, 'reviewText'] = df['reviewText'].str.lower()

# Instead of this:
# df['reviewText'] = df['reviewText'].apply(lambda x: re.sub('[^a-zA-Z 0-9]+', ' ', x))
df.loc[:, 'reviewText'] = df['reviewText'].apply(lambda x: re.sub('[^a-zA-Z 0-9]+', ' ', x))

df.head()


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from bs4 import BeautifulSoup

#removing special character
df['reviewText']=df['reviewText'].apply(lambda x:re.sub('[^a-zA-Z 0-9]+',' ',x))
#Remove Stopwords
df['reviewText']=df['reviewText'].apply(lambda x:" ".join([y for y in x.split() if y not in  stopwords.words('english') ]))
#Remove Url
df['reviewText']=df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))
## Remove html tags
df['reviewText']=df['reviewText'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
## Remove any additional spaces
df['reviewText']=df['reviewText'].apply(lambda x: " ".join(x.split()))

df.head()

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

def lemmatize_words(text):
  return" ".join([lemmatizer.lemmatize(word) for word in text.split()])

nltk.download('wordnet')
df['reviewText']=df['reviewText'].apply (lambda x:lemmatize_words(x))

df.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df['reviewText'],df['rating'],test_size=0.20)

from sklearn.feature_extraction.text import CountVectorizer
bow=CountVectorizer()
X_train_bow=bow.fit_transform(X_train).toarray()
X_test_bow=bow.transform(X_test).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X_train_tfidf=tfidf.fit_transform(X_train).toarray()
X_test_tfidf=tfidf.transform(X_test).toarray()

X_train_bow

from sklearn.naive_bayes import GaussianNB
nb_model_bow=GaussianNB().fit(X_train_bow,y_train)
nb_model_tfidf=GaussianNB().fit(X_train_bow,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

y_pred_bow=nb_model_bow.predict(X_test_bow)

y_pred_tfidf=nb_model_tfidf.predict(X_test_tfidf)

confusion_matrix(y_test,y_pred_bow)

print("BOW accuracy:",accuracy_score(y_test,y_pred_bow))

confusion_matrix(y_test,y_pred_tfidf)

print("TFIDF accuracy: ",accuracy_score(y_test,y_pred_tfidf))

# prompt: can u complete this code for sentiment analysis giving the sentiments of customers

print(classification_report(y_test,y_pred_bow))
print(classification_report(y_test,y_pred_tfidf))
def predict_sentiment(text, model, vectorizer):
    preprocessed_text = lemmatize_words(re.sub('[^a-zA-Z 0-9]+', ' ', text.lower()))
    features = vectorizer.transform([preprocessed_text]).toarray()
    sentiment = model.predict(features)[0]
    return 'Positive' if sentiment == 1 else 'Negative'

# Test the function with new text
new_text = "This book is an excellent read! I highly recommend it."
print("Sentiment (BoW):", predict_sentiment(new_text, nb_model_bow, bow))
print("Sentiment (TF-IDF):", predict_sentiment(new_text, nb_model_tfidf, tfidf))

new_text = "The product was defective and the customer service was terrible."
print("Sentiment (BoW):", predict_sentiment(new_text, nb_model_bow, bow))
print("Sentiment (TF-IDF):", predict_sentiment(new_text, nb_model_tfidf, tfidf))