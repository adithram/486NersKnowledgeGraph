# coding: utf-8
# ners.py
# Adapted from guide found here: https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/


# imports

import pandas as pd
import numpy as np
import sklearn
from sklearn_crfsuite import CRF
from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
from eli5 import show_weights

# loading dataset
training_data= pd.read_csv("data/ner_dataset.csv", encoding="latin1")
training_data = training_data.fillna(method="ffill")

# Peek at the data
training_data.tail(10)


# class for retrieving a sentence
# will create a tuple of word, pos, and tag
class RetrieveSentence(object):
    
    def __init__(self, data):
        self.sentence_num = 1
        self.data = data
        self.is_empty = False
        # Aggregating data
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.all_sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.sentence_num)]
            self.sentence_num += 1
            return s
        except:
            return None


retrieval = RetrieveSentence(training_data)
sentence = retrieval.get_next()

# sentence contrains a list of tuples, each tuple is the word, pos, and tag
print (sentence)


# retrieve all of the sentences as tuple
all_sentences = retrieval.all_sentences


# Feature Engineering
# Feature Engineering Strategy defined by Tobias Sterbak (https://www.depends-on-the-definition.com/about/)
def wordToFeature(sentence, i):
    word = sentence[i][0]
    postag = sentence[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sentence[i-1][0]
        postag1 = sentence[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sentence)-1:
        word1 = sentence[i+1][0]
        postag1 = sentence[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def convertToFeatures(sentence):
    return [wordToFeature(sentence, i) for i in range(len(sentence))]

def convertToLabels(sentence):
    return [label for token, postag, label in sentence]

def convertToTokens(sentence):
    return [token for token, postag, label in sentence]


features_vec = [convertToFeatures(sentence) for sentence in all_sentences]
labels = [convertToLabels(sentence) for sentence in all_sentences]

# instantiate CRF object
# L1 regularization parameter increased to improve focus on context
crf = CRF(algorithm='lbfgs',
          c1=10,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

# Make prediction on data with 5-folds cross validation
predictions = cross_val_predict(estimator=crf, X=features_vec, y=labels, cv=5)

# 5-folds cross validation report
cv_report = flat_classification_report(y_pred=predictions, y_true=labels)
print(cv_report)

crf.fit(features_vec, labels)

# Look at the weights
show_weights(crf, top=30)






