import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path

import random
import pickle
import json

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

file_path = Path(__file__).parent/"intents.json"
intents = json.load(open(file_path))
words = []
classes = []
documents = []
ignore_symbols = ['?', '!', ',', '.', ', ']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols]
words = sorted(set(words))

classes = sorted(set(classes))

folder = Path(__file__).parent/"model"

pickle.dump(words, open(folder/"words.pkl", 'wb'))
pickle.dump(classes, open(folder/"classes.pkl", 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word) for word in word_patterns if word not in ignore_symbols]

print(word_patterns)