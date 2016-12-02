import nltk
# import scikit
import pickle
import json
import sys
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


def getTokens(sentence):
    lowers = sentence.lower()
    no_punctuation = lowers.translate(string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

''' Training '''

with open('data/train.json') as rd:
  content = json.load(rd)

vectorizer = CountVectorizer(ngram_range=(1, 3))
stemmer = PorterStemmer()

print 'app Training..'
labels = []
train_features = vectorizer.fit_transform([r['text'] for r in content])

for review in content:
  if float(review['stars']) > 3:
    labels.append(1)
  else:
    labels.append(0)
print 'app Testng..'

''' Testing '''
with open('data/test.json') as rd:
  content = json.load(rd)

stemmer = PorterStemmer()
actual = []
test_features = vectorizer.transform([r['text'] for r in content])
for review in content:
  if float(review['stars']) > 3:
    actual.append(1)
  else:
    actual.append(0)

nb = MultinomialNB()
print 'Training..'
nb.fit(train_features, labels)

print 'Predicting..'
# Now we can use the model to predict classifications for our test features.
predictions = nb.predict(test_features)
print 'done'
target_names = ['0', '1']
# Compute the error.  It is slightly different from our model because the internals of this process work differently from our implementation.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
print(classification_report(actual, predictions, target_names=target_names))
print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))
