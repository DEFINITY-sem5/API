from flask import jsonify
import json
from newspaper import Article, Config
from keras.models import load_model
from keras.preprocessing.text import Tokenizer 
import pickle, re
from keras.preprocessing.sequence import pad_sequences
config = Config()
config.keep_article_html = True
tkn = Tokenizer()

def normalize(data):
    normalized = []
    for i in data:
        i = i.lower()
        #get rid of url's
        i = re.sub('https?://\S+|www\.\S+', '',i)
        #get rid of non words and extra spaces
        i = re.sub('\\W',' ',i)
        i = re.sub('\n',' ',i)
        i = re.sub(' +',' ',i)
        i = re.sub('^',' ',i)
        i = re.sub(' $', '',i)
        normalized.append(i)
    return normalized


def clickbait(title):
    cb_model = load_model("saved_models/Clickbait_savedmodel/clickbait_model.h5")
    with open("saved_models/Clickbait_savedmodel/clickbait_tokenizer.pickle", "rb") as cb_handle:
        cb_tkn = pickle.load(cb_handle)
    classes = ['non-clickbait','clickbait']

    input_text = str(title).lower()
    input_text = normalize(input_text) #cleaning text
    input_token = cb_tkn.texts_to_sequences([input_text])
    output = pad_sequences(input_token, padding='pre', maxlen=(15))
    processed_input = pad_sequences(output, padding='post', maxlen=(20))
    preds = cb_model.predict(processed_input)
    pred = preds.argmax(axis=-1)
    output = classes[pred[0]]
    return output

def fakenews(text):
    fn_model = load_model("saved_models/FakeNews_savedmodel/fakenews_model.h5")
    with open("saved_models/FakeNews_savedmodel/clickbait_tokenizer.pickle","rb") as fn_handle:
        fn_tkn = pickle.load(fn_handle)
    classes = ['legit news','fake news']
    input_text = str(text).lower()
    input_text = normalize(input_text) #cleaning text
    input_token = fn_tkn.texts_to_sequences([input_text])
    output = pad_sequences(input_token, padding='pre', maxlen=(256))
    processed_input = pad_sequences(output, padding='post', maxlen=(256))
    preds = fn_model.predict(processed_input)
    pred = preds.argmax(axis=-1)
    output = classes[pred[0]]
    return output

def extract(url):
    article = Article(url=url, config=config)
    article.download()
    article.parse()
    title = article.title
    text = article.text
    clickbait(title)
    fakenews(text)
    return title, text, clickbait(title),fakenews(text)
    

