from keras.optimizers import Nadam
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
import pandas as pd
import numpy as np
import random
import params
import sys

dt = pd.read_csv(params.re_file_path)
df = dt.sort_values(by='date')
rows, cols = df.shape
val = df.values[:, 1:6]

window_size = 100
ratio = 0.9
step = 10
x_size = int(window_size*ratio)
y_size = window_size - x_size

data_x_train = []
data_y_train = []

print('Sequences creating......')
for i in range(0, rows - window_size, step):
    data_x_train.append(val[i:i + x_size])
    data_y_train.append(val[i + x_size:i+window_size])

pattern_x = np.zeros((len(data_x_train), x_size), dtype=np.int32)
date_x = np.zeros((len(data_x_train), x_size, 2), dtype=float)

pattern_y = np.zeros((len(data_x_train), y_size), dtype=np.int32)
date_y = np.zeros((len(data_x_train), y_size), dtype=float)

print('Data sets preparing...')
for i, block in enumerate(data_x_train):
    for j, item_x in enumerate(block):
        pattern_x[i, j] = int(item_x[0])
        date_x[i, j, 0] = item_x[1]
        date_x[i, j, 0] = item_x[2]
    for k, item_y in enumerate(data_y_train[i]):
        pattern_y[i, k] = int(item_y[0])
        date_y[i, k] = item_y[1]

start_index = random.randint(0, len(data_x_train) - window_size)
examp = val[start_index: start_index + x_size]

date_pred = np.zeros((1, x_size, 2))
pattern_pred = np.zeros((1, x_size))

for m, item in enumerate(examp):
    date_pred[0, m, 0] = item[1]
    date_pred[0, m, 1] = item[2]
    pattern_pred[0, m] = item[0]

print("============test data:==========")
print(pattern_pred)
print(date_pred)
print("================================")

print('Model building...')
print(pattern_x[1:2, :])

pattern_input = Input(shape=(x_size,), dtype='int32', name='pattern_input')
date_input = Input(shape=(x_size, 2), dtype='float32', name='date_input')

x = Embedding(output_dim=128, input_dim=5, input_length=x_size)(pattern_input)

x = LSTM(64)(x)
d_y = LSTM(64)(date_input)
p_y = merge([x, d_y], concat_axis=1)
p_y = Dense(y_size, activation='relu', name='p_y')(p_y)
d_y = Dense(y_size, activation='relu', name='d_y')(d_y)
model = Model([pattern_input, date_input], [p_y, d_y])
optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit([pattern_x, date_x], [pattern_y, date_y], batch_size=128, nb_epoch=1)

predicts = model.predict([pattern_pred, date_pred], batch_size=128, verbose=1)
print(predicts)
