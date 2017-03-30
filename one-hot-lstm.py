# -*- coding: utf-8 -*-

# from keras.models import Sequential
# from keras.layers import Dense, Activation, Reshape
# from keras.layers import LSTM
# from keras.optimizers import Nadam
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


# def normalize(ls):
#     scale = 2*(max(ls)-min(ls))
#     mean = sum(ls)/len(ls)
#     return list(map(lambda n: (n-mean)/scale + 0.5, ls))

# val[:, 3] = normalize(val[:, 3])
# val[:, 4] = normalize(val[:, 4])

print('Sequences creating......')
for i in range(0, rows - window_size, step):
    data_x_train.append(val[i:i + x_size])
    data_y_train.append(val[i + x_size:i+window_size])

pattern_input = np.zeros((len(data_x_train), x_size, 5), dtype=np.bool)
date_input = np.zeros((len(data_x_train), x_size), dtype=float)
index_1 = np.zeros((len(data_x_train), x_size, 300), dtype=np.bool)
index_2 = np.zeros((len(data_x_train), x_size, 100), dtype=np.bool)
index_3 = np.zeros((len(data_x_train), x_size, 100), dtype=np.bool)

pattern_y = np.zeros((len(data_x_train), y_size, 5), dtype=np.bool)
date_y = np.zeros((len(data_x_train), y_size), dtype=float)
l = np.zeros((len(data_x_train), y_size, 300, 100, 100), dtype=np.bool)
# X = np.zeros((len(data_x_train), x_size, 5), dtype=float)
# # pattern = np.zeros((len(data_x_train), x_size, 5), dtype=np.bool)
# y = np.zeros((len(data_y_train), y_size, 5), dtype=float)

print('Data sets preparing...')
for i, block in enumerate(data_x_train):
    for j, item_x in enumerate(block):
        pattern_input[i, j, int(item_x[0])] = 1
        date_input[i, j] = item_x[1]
        user_num = int(item_x[3])
        i_1 = user_num / 10000
        i_2 = (user_num % 10000)/100
        i_3 = user_num % 1000000
        try:
            index_1[i, j, i_1] = 1
            index_2[i, j, i_2] = 1
            index_3[i, j, i_3] = 1
        except IndexError:
            continue
        # X[i, j] = item_x
        # pattern[i, j, int(item_x[0])] = 1
    for k, item_y in enumerate(data_y_train[i]):
        pattern_y[i, k, int(item_y[0])] = 1
        date_y[i, k] = item_y[1]
        i_1 = user_num / 10000
        i_2 = (user_num % 10000) / 100
        i_3 = user_num % 1000000
        try:
            l[i, k, i_1, i_2, i_3] = 1
        except IndexError:
            continue
        # y[i, k] = item_y

print('Model building...')

pattern_x = Input(shape=(x_size, 5), dtype=np.bool, name='pattern_input')
date_x = Input(shape=(x_size, ), dtype=float, name='date_input')
f1 = Input(shape=(x_size, 300), dtype=np.bool, name='f1')
f2 = Input(shape=(x_size, 100), dtype=np.bool, name='f2')
f3 = Input(shape=(x_size, 100), dtype=np.bool, name='f3')

pattern_x = LSTM(128)(pattern_x)
date_x = LSTM(128)(date_x)
f1 = Dense(y_size*300, activation='relu')(f1)
f2 = Dense(y_size*100, activation='relu')(f2)
f3 = Dense(y_size*100, activation='relu')(f3)
f = merge([f1, f2, f3], mode='concat')
pattern_x = merge([pattern_x, date_input, f], mode='concat')
pattern_y = Dense(128, activation='softmax')(pattern_x)

model = Model(input=[pattern_input, date_input, f1, f2, f3], output=pattern_y)
model.compile(optimizer='nadam', loss='categorical_crossentryopy', metrics=['accuracy'])
model.fit([pattern_input, date_input, index_1, index_2, index_3], pattern_y, nb_epoch=50, batch_size=32)
# model = Sequential()
#
# model.add(LSTM(128, input_shape=(x_size,5)))
# model.add(Dense(y_size*(cols-1)))
# model.add(Activation('softmax'))
# model.add(Reshape((y_size, cols-1)))
#
# optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def sample(predicts, temperature):
    predicts = np.asarray(predicts).astype('float64')
    predicts = np.log(predicts) / temperature
    exp = np.exp(predicts)
    predicts = exp / np.sum(exp)
    alpha = np.random.multinomial(1, predicts)
    return np.argmax(alpha)


for iteration in range(1, 100):
    print()
    print('Iteration', iteration)
    model.fit([pattern_x, date_x, f1, f2, f3], [pattern_y, date_y, l1, l2, l3], batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(data_x_train) - window_size)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('diversity: ', diversity)

        generated = []
        examp = val[start_index: start_index + x_size]
        generated.append(examp)

        for i in range(400):
            x = np.zeros((1, x_size, 5))
            for j, item in enumerate(examp):
                x[0, j] = examp[i]

            print(x)
            print(model.predict(x, verbose=0))

            predicts = model.predict(x, verbose=0)[0]
            print('predictions: ', predicts)
            next_index = sample(predicts, diversity)
            next_res = []
            for index in next_index:
                next_res.append(val[index])

            generated.append(next_res)
            examp = examp[y_size:] + next_res

            sys.stdout.write(examp)
            sys.stdout.flush()
        print()






