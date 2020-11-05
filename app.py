from flask import Flask
from newspaper import Article
import numpy
from threading import Thread
import urllib3
import time

app = Flask(__name__)

model_cb = None
model_fn = None

urllib3.disable_warnings()
np.warnings.filterwarnings('ignore')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')