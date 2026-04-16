import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path

import random
import pickle
import json

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

folder = Path(__file__).parent/"model"

def sentence_cleaner(sentence):
    ignore_symbols = ['?', '!', ',', '.', ', ']
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize()
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in ignore_symbols]
    return sentence_words

def bag_of_words(sentence):
    
    words = pickle.load(open(folder/'words.pkl', 'rb'))

    sentence_words = sentence_cleaner(sentence)
    bag = [0]*len(words)

    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w: 
                bag[i]=1

    return np.array(bag)

def predict_class(sentence):
    classes = pickle.load(open(folder/'classes.pkl', 'rb'))
    model = load_model(folder/'chatbot_model.keras')

    