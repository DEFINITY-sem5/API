import flask
from newspaper import Article
from keras.models import load_model
import functools

app = flask.Flask(__name__)

model_cb = load_model("saved_models/Clickbait_savedmodel/clickbait_model.h5")
model_fn = load_model("saved_models/FakeNews_savedmodel/fakenews_model.h5")

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
    if flask.request.method == "POST":
        url = flask.request.args.get("url")
        article_title, article_text = read(url)




if __name__ == "__main__":
    print("Starting Flask server")
    app.run(host='0.0.0.0', port=4000)