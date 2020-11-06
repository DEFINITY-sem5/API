import numpy as np
from newspaper import Article
import re
import functools
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

def pred_clickbait(input_):
    global model_clickbait, data
    data["clickbait"] = model_cb.predict(input_)

def pred_fakenews(input_):
    global model_profile, data
    data["article_profile"] = model_fn.predict(input_) 

urllib3.disable_warnings()
np.warnings.filterwarnings('ignore')


data = {"success": False}
#to look into URL's
def read(url):
    article = Article(url, fetch_images = False)
    article.download()
    article.parse()
    article_title = article.title
    article_text = article.text
    return(article_title, article_text)


@functools.lru_cache(maxsize=512, typed=False)
@app.route("/predict", methods=["POST"])

def predict():

    global model_cb, model_fn
    global data

    data = {"success" : False}

    if flask.request.method == "POST":
        #url = flask.request.args.get("url")
        url = "https://timesofindia.indiatimes.com/city/mumbai/no-evidence-to-show-juhu-cops-intended-to-cause-death-court/articleshow/79074605.cms"
        article_title, article_text = read(url)

        threads=[]

        if article_title is not None:
            t = Thread(target=pred_fakenews, args=([article_text]))
            threads.append(t)
            t.start
        
        if article_text is not None:
            article_title = article_title.replace("%20", " ")
            data["article_text"] = article_title

            t = Thread(target=pred_clickbait, args=([article_title]))
            threads.append(t)
            t.start


        [t.join() for t in threads]
        data["success"] = True

    return flask.jsonify(data)

if __name__ == "__main__":
    print("Starting Flask server")
    app.run(host='0.0.0.0', port=4000)
