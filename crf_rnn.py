# coding: utf-8
# crf_nn.py
# Adapted from guide found here: https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/


# imports

import pandas as pd
import numpy as np
# import sklearn
# from sklearn_crfsuite import CRF
# from sklearn.cross_validation import cross_val_predict
# from sklearn_crfsuite.metrics import flat_classification_report
# import eli5
import nltk
# import pickle
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Character Embedding Model Imports
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D

# Plitting imports
import matplotlib.pyplot as plt






# Read the data
# https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
training_data = pd.read_csv("data/ner_dataset.csv", encoding="latin1")
training_data = training_data.fillna(method="ffill")


# Get the respective words and tags
words = list(set(training_data["Word"].values))
n_words = len(words); n_words

tags = list(set(training_data["Tag"].values))
n_tags = len(tags); n_tags

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

# Helper functions used to pull data from training set and convert to desired format
def convertToFeatures(sentence):
    return [wordToFeature(sentence, i) for i in range(len(sentence))]

def convertToLabels(sentence):
    return [label for token, postag, label in sentence]

def convertToTokens(sentence):
    return [token for token, postag, label in sentence]


# Instantiate previously used sentence retrieval class
# class for retrieving a sentence
# will create a tuple of word, pos (part of speech), and tag (NER tag)
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


# Retrieve sentence
# create retrieval object
retrieval = RetrieveSentence(training_data)
# sentence contrains a list of tuples, each tuple is the word, pos, and tag

# retrieve all of the sentences as tuples
all_sentences = retrieval.all_sentences

# Token preparization 

# Arbitrary max
max_len = 75
max_len_char = 10

# ALl for padding purposes
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0
idx2word = {i: w for w, i in word2idx.items()}
tag2idx = {t: i + 1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}

# Now we map the sentences to a sequence of numbers and then pad the sequence. 
# Note that we increased the index of the words by one to use zero as a padding value. 
# This is done because we want to use the mask_zero parameter of the embedding layer to ignore inputs with value zero.

# convert words to numeric values from training data with padding
X_word = [[word2idx[w[0]] for w in s] for s in all_sentences]

# pad_sequences function from keras
X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')

chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)

char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

# convert chars in training data to numeric values with padding
X_char = []
for sentence in all_sentences:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char.append(np.array(sent_seq))


# Match training data labels to chars
y = [[tag2idx[w[2]] for w in s] for s in all_sentences]

# Training data labels with matching padding
y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

# Split the data
X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)
X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2018)


# RNN/LSTM LAYERS - Check Keras Documentation as needed
# #########################################################################################################################

# input and embedding for words
word_in = Input(shape=(max_len,))
emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                     input_length=max_len, mask_zero=True)(word_in)

# input and embeddings for characters
char_in = Input(shape=(max_len, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                           input_length=max_len_char, mask_zero=True))(char_in)
# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                recurrent_dropout=0.5))(emb_char)

# main LSTM
x = concatenate([emb_word, char_enc])
x = SpatialDropout1D(0.3)(x)
main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.6))(x)
out = TimeDistributed(Dense(n_tags + 1, activation="sigmoid"))(main_lstm)

# Final model
model = Model([word_in, char_in], out)
# #########################################################################################################################

# Compile model
# Using the adam optimizer, sparse categorical crossentropy loss
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

# Print the model summary
# model.summary()

# Fit the model
history = model.fit([X_word_tr, np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))], 
	np.array(y_tr).reshape(len(y_tr), max_len, 1),
	batch_size=32, epochs=10, validation_split=0.1, verbose=1)

# # Dumping to pickle file 
# pickle.dump(model, open('final_model.sav', 'wb'), protocol=2)

# print("Data dumped to pickle file")

############################################################################################
## Validation and Performance Evaluation

# hist = pd.DataFrame(history.history)

# # Plotting
# plt.style.use("ggplot")
# plt.figure(figsize=(12,12))
# plt.plot(hist["acc"])
# plt.plot(hist["val_acc"])
# plt.show()

# # Retrieve predictions (on test set)
# y_pred = model.predict([X_word_te, np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))])

# # Print predictions (on test set)
# i = 1925
# p = np.argmax(y_pred[i], axis=-1)
# print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
# print(30 * "=")
# for w, t, pred in zip(X_word_te[i], y_te[i], p):
#     if w != 0:
#         print("{:15}: {:5} {}".format(idx2word[w], idx2tag[t], idx2tag[pred]))



############################################################################################
# Predict on input data 

# Read raw text
unprocessed_text = ''
with open("text_files/rawtext.txt", "r") as raw:
    unprocessed_text = raw.read()

# split raw text into list of sentences
raw_sentences = []
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(unprocessed_text)

# list of tagged sentences
tagged_sentences = []
for sentence in raw_sentences:
    text = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(text)
    tagged_sentences.append(tagged)


# Char vector for input data
raw_chars = []
for sentence in tagged_sentences:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    raw_chars.append(np.array(sent_seq))

# Word vector for input data
raw_words = [[word2idx[w[0]] for w in s] for s in tagged_sentences]
raw_words = pad_sequences(maxlen=max_len, sequences=raw_words, value=word2idx["PAD"], padding='post', truncating='post')

preds = model.predict([raw_words, np.array(raw_chars).reshape((len(raw_chars), max_len, max_len_char))])

####################################################################################
# Sequential named entitites are not grouped

# outputting
named_entities = []
tagged_named_entities = []
for i, sentence in enumerate(tagged_sentences):
    for j, word in enumerate(sentence):
        # if the word in the sentence is a named entity:
        if preds[i][j] != 'O':
            # print(word[0])
            named_entities.append(word[0])
            tmp = (word[0], preds[i][j])
            tagged_named_entities.append(tmp)

with open('text_files/named_entities.txt', 'w') as f:
    for entity in named_entities:
        f.write(entity +'\n')

with open('text_files/tagged_named_entities.txt', 'w') as f:
    for entity in tagged_named_entities:
        f.write(entity[0] + ': ' + entity[1]  +'\n')



####################################################################################
# Group sequential named entitites together
last_index = 0
combined_named_entities = []
for i, sentence in enumerate(tagged_sentences):
    j = 0 
    while j < len(sentence):
    # for j, word in enumerate(sentence):
        combined_indices = [j]
        if preds[i][j] != 'O':
            
            while preds[i][j+1] != 'O':
                j+= 1
                combined_indices.append(j)
                # last_index = c

            w = ''
            for num in combined_indices:
                w += sentence[num][0] + ' '
            combined_named_entities.append(w)
        j+=1

with open('text_files/combined_named_entities.txt', 'w') as f:
    for entity in combined_named_entities:
        f.write(entity +'\n')


