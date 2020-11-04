from model_func import *

import tensorflow as tf 
import functionals
import re
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences