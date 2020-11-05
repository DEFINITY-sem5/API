import numpy as np
from newspaper import Article
import re
import flask
import urllib3
from threading import Thread
from classifier import *

#app 
app = flask.Flask(__name__)

model_cb = None
model_fn = None

def Load_ML():
    global model_cb, model_fn
    model_fn = Fakenews_classifier()
    model_cb = Clickbait_classifier()

Load_ML()

urllib3.disable_warnings()
np.warnings.filterwarnings('ignore')


#to look into URL's
def read(url):
    article = Article(url, fetch_images = False)
    article.download()
    article.parse()
    article_title = article.title
    article_text = article.text
    return(article_title, article_text)

@app.route("/predict", methods=["POST"])

def predict():
    if flask.request.method == "POST":
        url = flask.request.args.get("url")
        article_title, article_text = read(url)

        
        
if __name__ == "__main__":
    print(" * [i] Starting Flask server")
    app.run(host='0.0.0.0', port=5000)