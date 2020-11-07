from newspaper import Article, Config
from keras.models import load_model
from keras.preprocessing.text import Tokenizer 
import pickle, re
from keras.preprocessing.sequence import pad_sequences
config = Config()
config.keep_article_html = True
tkn = Tokenizer()

def clickbait(title):
    cb_model = load_model("saved_models/Clickbait_savedmodel/clickbait_model.h5")
    with open("saved_models/Clickbait_savedmodel/clickbait_tokenizer.pickle", "rb") as cb_handle:
        cb_tkn = pickle.load(cb_handle)
    classes = ['non-clickbait','clickbait']

    input_text = str(title).lower()
    input_text = re.sub(r'[^\w\s]','',input_text) #cleaning text
    input_token = cb_tkn.texts_to_sequences([input_text])
    output = pad_sequences(input_token, padding='pre', maxlen=(15))
    processed_input = pad_sequences(output, padding='post', maxlen=(20))
    preds = cb_model.predict(processed_input)
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
    return clickbait(title)