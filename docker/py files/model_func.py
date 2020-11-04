import pickle
import numpy as np
import pandas as pd

import keras 
from keras.layers import *
from keras.models import Model, load_model
from keras import backend as bnd

class api_model(object):
    def __init__(self, debug=True):
        self.name = "model's name"
        self.debug = debug 

        if self.debug:
            if self.run_self_tests():
                print("*[i] Model: " + self.name + " has loaded successfully")
            else:
                print("* [!] An error has occured in self test!") 

    def run_self_tests(self):
        #leave in a simple test to see if the model runs
        # also to take a quick benchmark to test performance
        return True
    
    def pred(self, input_data):
        # wrap the model.predict function here
        # it is a good idea to just do the pre-processing here also
        return NotImplementedError

    def preprocess(self, input_data):
        # preprocessing function
        return NotImplementedError