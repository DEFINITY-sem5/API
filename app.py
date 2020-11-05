from flask import Flask
from newspaper import Article
import numpy
from threading import Thread

app = Flask(__name__)

@app.route('/')
def hello_world():
    return("Build Commencing")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')