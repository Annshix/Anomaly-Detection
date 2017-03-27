# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import LSTM
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import random
import params
import sys


class LstmProcessing:
    def __init__(self, trained):
        self.dt = pd.read_csv(params.re_file_path)
        self.rows, self.cols = self.dt.shape
        self.val = self.dt.values[:, 1:6]
        self.x_size = int(params.window_size * params.ratio)
        self.y_size = params.window_size - self.x_size
        self.data_x_train = []
        self.data_y_train = []
        self.trained = trained
        self.model = Sequential()
        self.__model_building()

    def data_prepare(self):
        if self.trained:
            return

        print('Sequences creating......')
        for i in range(0, self.rows - params.window_size, params.step):
            self.data_x_train.append(self.val[i:i + self.x_size])
            self.data_y_train.append(self.val[i + self.x_size:i + params.window_size])

        X = np.zeros((len(self.data_x_train), self.x_size, 5), dtype=float)
        y = np.zeros((len(self.data_y_train), self.y_size, 5), dtype=float)

        print('Data sets preparing...')
        for i, block in enumerate(self.data_x_train):
            for j, item_x in enumerate(block):
                X[i, j] = item_x
            for k, item_y in enumerate(self.data_y_train[i]):
                y[i, k] = item_y

        return [X, y]

    def __model_building(self):
        if self.trained:
            return
        print('Model building...')
        self.model.add(LSTM(128, input_shape=(self.x_size, self.cols - 1)))
        self.model.add(Dense(self.y_size * (self.cols - 1)))
        self.model.add(Activation('softmax'))
        self.model.add(Reshape((self.y_size, self.cols - 1)))

        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    @staticmethod
    def sample(predicts, temperature):
        predicts = np.asarray(predicts).astype('float64')
        predicts = np.log(predicts) / temperature
        exp = np.exp(predicts)
        predicts = exp / np.sum(exp)
        alpha = np.random.multinomial(1, predicts)
        return np.argmax(alpha)

    def model_fit(self):
        for iteration in range(1, 100):
            print()
            print('Iteration', iteration)
            X, y = self.data_prepare()
            if not self.trained:
                self.model.fit(X, y, batch_size=128, nb_epoch=1)

            start_index = random.randint(0, len(self.data_x_train) - params.window_size)

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print()
                print('diversity: ', diversity)

                generated = []
                examp = self.val[start_index: start_index + self.x_size]
                generated.append(examp)

                for i in range(400):
                    x = np.zeros((1, self.x_size, 5))
                    for j, item in enumerate(examp):
                        x[0, j] = examp[i]

                    predicts = self.model.predict(x, verbose=0)[0]
                    print('predictions: ', predicts)
                    next_index = self.sample(predicts, diversity)
                    next_res = []
                    for index in next_index:
                        next_res.append(self.val[index])

                    generated.append(next_res)
                    examp = examp[self.y_size:] + next_res

                    sys.stdout.write(examp)
                    sys.stdout.flush()
                print()
