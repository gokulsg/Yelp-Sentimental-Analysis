# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:31:35 2016

@author: mounik
"""

import sys
import json 
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from stemming.porter2 import stem
import time

def LoadDataFromFile(filename):
    # This function is used to load 5000 reviews of positve and 5000 reviews of negative
    with open(filename) as infile:
        features = []
        labels = []
        num_data = 0
        num_pos_data = 0
        num_neg_data = 0
        data_size = 5000
        half_size = data_size / 2
        for line in infile:
            review = json.loads(line)
            if int(review['stars']) > 3:    # Positive
                if num_pos_data < half_size:
                    labels.append(1)
                    features.append(review['text'])
                    num_pos_data += 1
                    num_data += 1

            if int(review['stars']) < 3:    # Negative
                if num_neg_data < half_size:
                    labels.append(0)
                    features.append(review['text'])
                    num_neg_data += 1
                    num_data += 1

            if num_pos_data + num_neg_data == data_size:
                # features = features_pos[0:1000] + features_neg[0:1000]
                print (len(features), len(labels), num_pos_data, num_neg_data)
                return features, labels
    return features, labels

def StemWords(X):
    # This function performs stemming in words
    X_stem_words = []
    for review in X:
        review = str(review)
        review = [stem(word) for word in review.split(" ")]
        reviews = ' '.join(review)
        X_stem_words.append(reviews)
    return X_stem_words


def PerformLogisticRegression(X_train,y_train, X_test, y_test):
    # This function performs Logistic Regression and displays the results
    logReg = linear_model.LogisticRegression(penalty='l2', max_iter = 100, solver='newton-cg')
    logReg.fit(X_train, y_train)
    y_pred = logReg.predict(X_test)
    print("Logistic regression\n%s\n" % (
    metrics.classification_report(
        y_test,
        y_pred)))
    print("Accuracy of algorithm: "+str(metrics.accuracy_score(y_test,y_pred)*100))

if __name__ == "__main__":
    start_time = time.time()
    y_test = []
    y_train = []
    X = []
    y = []
    filename = str(sys.argv[1]) # path of the file of reviews
    X, y = LoadDataFromFile(filename)
    # Stemming and Lemmatization of reviews
    #X = StemWords(X)
    # Spliting the databases into training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=42)
    # Extracting features using CountVectorizer
    count_vectorizer = CountVectorizer(max_features = 4000)
    count_train_vectors = count_vectorizer.fit_transform(X_train)
    count_test_vectors = count_vectorizer.fit_transform(X_test)
    # Implement logistic regression
    print("-----------Logisitc Regression using countvectorizer-------------")
    PerformLogisticRegression(count_train_vectors, y_train, count_test_vectors, y_test)
    # Extracting features using Tf-idf Vectorizer
    print("-----------Logisitc Regression using Tfidf vectorizer-------------")
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)
    tf_idf_train_vectors = vectorizer.fit_transform(X_train)
    tf_idf_test_vectors = vectorizer.transform(X_test)
    # Implement Logistic Regression
    PerformLogisticRegression(tf_idf_train_vectors, y_train, tf_idf_test_vectors, y_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    