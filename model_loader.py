import tensorflow as tf 
from tensorflow import keras
from model_base import *
import re
import pickle
import time
import functools
from nltk.corpus import stopwords
from keras.models import load_model
from keras.preprocessing.text import Tokenizer as tkn 
from keras.preprocessing.sequence import pad_sequences

class cb_classifier(api_model):
    def __init__(self, debug = False):
        self.name = "CLICKBAIT MODEL"
        self.model = load_model("saved_models/Clickbait_savedmodel/clickbait_model.h5")
        self.model._make_predict_function()
        with open("saved_models/Clickbait_savedmodel/clickbait_tokenizer.pickle", 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.classes = ["not_clickbait", "clickbait"]
        self.debug = debug
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name +
                      " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    def run_self_test(self):
        print(" * [i] Performing self-test...")
        try:
            # warm-up run
            test_string = self.preprocess(
                "32 ways to test a server. You won't believe no. 3!")
            self.model.predict(test_string)
            # benchmark run
            start = time.time()
            test_string = self.preprocess(
                "99 ways to wreck a paper. You will believe no. 4!")
            self.model.predict(test_string)
            print(" * [i] Server can process ", round(1 /
                                                      (time.time()-start), 1), "predictions per second")
            return True
        except Exception as e:
            print(" * [!] An error has occured:")
            print(e)
            return False

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



class fn_classifier(api_model):

    def __init__(self, debug=True):
        self.name = "Fake News Classifier"
        self.debug = debug
        self.model = load_model("saved_models/FakeNews_savedmodel/fakenews_model.h5")
        self.model._make_predict_function()
        self.classes = ["fake", "reliable"]
        with open('saved_models/FakeNews_savedmodel/clickbait_tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model: " + self.name +
                      " has loaded successfully")
            else:
                print(" * [!] An error has occured in self test!!")

    def run_self_test(self):
        input_sequence = self.preprocess(
            ["Hello there this is to cache the stops words I think not really sure why else got error when threaded lmao"])
        self.model.predict(input_sequence)
        return True

    @functools.lru_cache(maxsize=512, typed=False)
    def predict(self, input_data):
        input_sequence = self.preprocess(input_data)
        preds = self.model.predict(input_sequence)
        pred = preds.argmax(axis=-1)
        output = self.classes[pred[0]]
        return output

    def clean_text(self, text):
        output = ""
        text = str(text).replace("\n", "")
        text = re.sub(r'[^\w\s]', '', text).lower().split(" ")
        for word in text:
            if word not in stopwords.words("english"):
                output = output + " " + word
        return output.strip().replace("  ", " ")

    def preprocess(self, input_data):
        input_string = self.clean_text(input_data)
        input_token = self.tokenizer.texts_to_sequences([input_string])
        processed_input = pad_sequences(
            input_token, padding='post', maxlen=(200))
        return processed_input
