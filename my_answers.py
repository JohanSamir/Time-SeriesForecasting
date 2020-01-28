import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# DONE: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    count = len(series) - window_size

    for x_idx in range(0, count):
        y_idx = x_idx + window_size
        X.append(series[x_idx:y_idx])
        y.append(series[y_idx])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))

    return model

### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    letters = set(map(chr, range(ord('a'), ord('z')+1))) #a-z

    def is_valid(char):
        return char in letters or char in punctuation or char == ' '

    text = text.lower()
    text = filter(is_valid, text)
    text = ''.join(text)

    return text

### DONE: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    count = len(text) - window_size

    for in_idx in range(0, count, step_size):
        out_idx = in_idx + window_size
        inputs.append(text[in_idx:out_idx])
        outputs.append(text[out_idx])

    return inputs,outputs

# DONE build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()

    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))

    return model
