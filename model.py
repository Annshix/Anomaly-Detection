# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import random
import params

dt = pd.read_csv(params.re_file_path)
rows, cols = dt.shape
val = dt.values

window_size = 100
ratio = 0.7
step = 20
x_size = window_size*ratio
y_size = window_size*(1-ratio)

data_x_train = []
data_y_train = []

print('Sequences creating......')
for i in range(0, rows-window_size, step):
    data_x_train.append(val[i:i + x_size])
    data_y_train.append(val[i + x_size:i+window_size])

X = np.zeros((len(data_x_train), x_size, 5), dtype=float)
y = np.zeros((len(data_y_train), y_size, 5), dtype=float)

print('Data sets preparing...')
for i, block in enumerate(data_x_train):
    for j, item_x in enumerate(block):
        X[i, j] = item_x
    for k, item_y in enumerate(data_y_train[i]):
        y[i, k] = item_y

print('Model building...')

model = Sequential()

model.add(LSTM(200, input_shape=(window_size, cols)))
model.add(Dense(cols))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample():
    for iteration in range(1, 100):
        print()
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('diversity: ', diversity)

            generated = []

            for i in range(400):
                x = np.zeros



