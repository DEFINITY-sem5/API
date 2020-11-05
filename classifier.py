import re
from nltk.corpus import stopwords
import tensorflow as tf
import keras 
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from base import *

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
