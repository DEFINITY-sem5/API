import re
from nltk.corpus import stopwords
import tensorflow as tf
import keras 
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Fakenews_classifier():
    def __init__(debug=True):
        name = "Fake News Classifier"
        fn_model = load_model("saved_models/FakeNews_savedmodel/fakenews_model.h5")
        classes = ['Fake', 'Reliable']
        with open('saved_models/FakeNews_savedmodel/clickbait_tokenizer.pickle', 'rb') as fn_handle:
            fn_tkn = pickle.load(fn_handle)


class Clickbait_classifier():
    def __init__(debug=True):
        name = "Clickbait Classifier"
        cb_model = load_model("saved_models/Clickbait_savedmodel/clickbait_model.h5")
        classes = ['clickbait', 'not-clickbait']
        with open("saved_models/Clickbait_savedmodel/clickbait_tokenizer.pickle",'rb') as cb_handle:
            cb_tkn = pickle.load(cb_handle)
