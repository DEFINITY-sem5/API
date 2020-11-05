import time
import numpy as np
import flask
import urllib3
from newspaper import Article
from threading import Thread

from model_loader import *

app = flask.Flask(__name__)

model_cb = None
model_fn = None

urllib3.disable_warnings()
np.warnings.filterwarnings('ignore')


def load_ML():
    global model_cb, model_fn

    model_cb = cb_classifier()
    model_fn = fn_classifier()

load_ML()

def pred_clickbait(input_):
    global model_cb, data
    data["clickbait"] = model_cb.predict(input_)

def pred_profile(input_):
    global model_fn, data
    data["article_profile"] = model_fn.predict(input_)

data = {"success": False}

@functools.lru_cache(maxsize=512, typed=False)
def download_article(article_url):
article = Article(article_url, fetch_images=False)
article.download()
article.parse()
article_title = article.title
article_text = article.text
image_list = article.images
return article_title, article_text, image_list
@app.route("/predict", methods=["POST"])
def predict():
# initialize the data dictionary that will be returned from the
# view
global model_clickbait, model_profile, model_subj
global data, cr
data = {"success": False}
# get the respective args from the post request
if flask.request.method == "POST":
start_time = time.time()
article_url = flask.request.args.get("article_url")
article_title, article_text, image_list = download_article(article_url)
article_time = time.time()
print(" * [i] Article download time:", round(article_time-start_time, 3), "seconds")
threads = []
if article_text is not None:
t = Thread(target=pred_profile, args=([article_text]))
threads.append(t)
t.start()
t = Thread(target=pred_subj, args=([article_text]))
threads.append(t)
t.start()
if article_title is not None:
article_title = article_title.replace("%20", " ")
print(" * [i] Incoming article title:", article_title)
data["article_title"] = article_title
t = Thread(target=pred_clickbait, args=([article_title]))
threads.append(t)
t.start()
data["claimReview"] = cr.search_fc(article_title)
if image_list is not None:
results = []
# TODO
data["hoax_image_search"] = results
        [t.join() for t in threads]
data["success"] = True
print(" * [i] Inference took", round(time.time()-article_time, 3), "seconds")
print(" * [i] Request took", round(time.time()-start_time, 3), "seconds")
# return the data dictionary as a JSON response
return flask.jsonify(data)
# if file was executed by itself, start the server process
if __name__ == "__main__":
print(" * [i] Starting Flask server")
app.run(host='0.0.0.0', port=5000)