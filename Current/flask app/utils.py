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

folder = Path(__file__).parent / "model"

lemmatizer = WordNetLemmatizer()

# Load once (IMPORTANT optimization)
words = pickle.load(open(folder / 'words.pkl', 'rb'))
classes = pickle.load(open(folder / 'classes.pkl', 'rb'))
model = load_model(folder / 'chatbot_model.keras')


def sentence_cleaner(sentence):
    ignore_symbols = ['?', '!', ',', '.', ', ']
    
    sentence_words = nltk.word_tokenize(sentence.lower())  # lowercase fix
    sentence_words = [
        lemmatizer.lemmatize(word)
        for word in sentence_words
        if word not in ignore_symbols
    ]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = sentence_cleaner(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]

    ERROR_THRESHOLD = 0.1   # lowered threshold

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({
            'intent': classes[r[0]],
            'probability': str(r[1])
        })

    return return_list   # ✅ FIXED (this was missing)


def get_response(intents_list):
    if not intents_list:
        return "I didn't understand that."

    intents_json = json.load(
        open(Path(__file__).parent.parent / 'model training' / 'intents.json')
    )

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return "No response found."