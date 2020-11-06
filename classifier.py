import re
from nltk.corpus import stopwords
import tensorflow as tf
import keras 
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from base import *
import functools

class Fakenews_classifier(api_model):
    def __init__(self, debug=True):
        self.name = "Fake News Classifier"
        self.debug = debug
        self.model = load_model("saved_models/FakeNews_savedmodel/fakenews_model.h5")
        self.classes = ['Fake', 'Reliable']
        with open('saved_models/FakeNews_savedmodel/clickbait_tokenizer.pickle', 'rb') as fn_handle:
            self.tkn = pickle.load(fn_handle)
        

class Clickbait_classifier():
    def __init__(self, debug=True):
        self.name = "Clickbait Classifier"
        self.debug = debug
        self.model = load_model("saved_models/Clickbait_savedmodel/clickbait_model.h5")
        self.classes = ['clickbait', 'not-clickbait']
        with open("saved_models/Clickbait_savedmodel/clickbait_tokenizer.pickle",'rb') as cb_handle:
            self.tkn = pickle.load(cb_handle)
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name +
                      " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    @functools.lru_cache(maxsize=512, typed=False)
    def predict(self, input_data):
        processed_input = self.preprocess(input_data)
        preds = self.model.predict(processed_input)
        pred = preds.argmax(axis=-1)

        output = self.classes[pred[0]]

        if self.debug:
            print(output)

        return output

    def preprocess(self, input_data):
        input_string = str(input_data).lower()
        input_string = re.sub(r'[^\w\s]', '', input_string)

        input_token = self.tokenizer.texts_to_sequences([input_string])
        output_t = pad_sequences(input_token, padding='pre', maxlen=(15))
        processed_input = pad_sequences(output_t, padding='post', maxlen=(20))

        if self.debug:
            print(" * [d] Cleaned string", input_string)
            print(" * [d] Test sequence", processed_input)

        return processed_input    