# -*- coding: utf-8 -*-


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import pandas as pd
import numpy as np
import random
import params

dt = pd.read_csv(params.re_file_path)
row, col = dt.shape

window_size = 100
ratio = 0.7
step = 5

X = np.zeros(window_size, col, dtype=np.bool)
y = np.zeros(ratio * window_size, col, dtype=np.bool)

print('starting...')

model = Sequential()

model.add(LSTM(200, input_shape=(window_size, col)))
model.add(Dense(col))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

