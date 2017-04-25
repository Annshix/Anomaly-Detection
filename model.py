from keras.optimizers import Nadam
from keras.layers import Input, LSTM, Dense, merge, Reshape
from keras.models import Model
import pandas as pd
import numpy as np
import random
import params


class ModelBuilding:
    def __init__(self):
        print('Initiating......')
        dt = pd.read_csv(params.train_file_path).sort_values(by='date')
        self.rows, self.cols = dt.shape
        self.val = dt.values[:, 1:6]
        self.window_size = 100
        ratio = 0.9
        self.step = 20
        self.x_size = int(self.window_size*ratio)
        self.y_size = self.window_size - self.x_size

        self.data_x_train, self.data_y_train = [], []
        for i in range(0, self.rows - self.window_size, self.step):
            self.data_x_train.append(self.val[i:i + self.x_size])
            self.data_y_train.append(self.val[i + self.x_size:i + self.window_size])

        self.pattern_x = np.zeros((len(self.data_x_train), self.x_size, 5), dtype=float)
        self.date_x = np.zeros((len(self.data_x_train), self.x_size, 2), dtype=float)
        self.user_x = np.zeros((len(self.data_x_train), self.x_size, 5), dtype=float)
        self.item_x = np.zeros((len(self.data_x_train), self.x_size, 5), dtype=float)

        self.pattern_y = np.zeros((len(self.data_x_train), self.y_size, 5), dtype=float)
        self.date_y = np.zeros((len(self.data_x_train), self.y_size, 2), dtype=float)
        self.user_y = np.zeros((len(self.data_x_train), self.y_size, 5), dtype=float)
        self.item_y = np.zeros((len(self.data_x_train), self.y_size, 5), dtype=float)

        self.test_num = 100
        self.date_pred = np.zeros((self.test_num, self.x_size, 2), dtype=float)
        self.pattern_pred = np.zeros((self.test_num, self.x_size, 5), dtype=float)
        self.user_pred = np.zeros((self.test_num, self.x_size, 5), dtype=float)
        self.item_pred = np.zeros((self.test_num, self.x_size, 5), dtype=float)
        self.pattern_truth = np.zeros((self.test_num, self.y_size), dtype=int)

        self.__data_prepare()
        self.__data_predict()

    def __data_prepare(self):
        print('Data sets preparing...')
        for i, block in enumerate(self.data_x_train):
            for j, item in enumerate(block):
                self.pattern_x[i, j, int(item[0])] = 1
                self.date_x[i, j, 0] = item[1]
                self.date_x[i, j, 1] = item[2]
                self.user_x[i, j, int(item[3] - 1)] = 1
                self.item_x[i, j, int(item[4] - 1)] = 1
            for k, item in enumerate(self.data_y_train[i]):
                self.pattern_y[i, k, int(item[0])] = 1
                self.date_y[i, k, 0] = item[1]
                self.date_y[i, k, 1] = item[2]
                self.user_y[i, k, int(item[3] - 1)] = 1
                self.item_y[i, k, int(item[4] - 1)] = 1

    def __data_predict(self):
        print('Data to predict......')
        start_index = []
        examp = []
        for i in range(self.test_num):
            start_index.append(random.randint(0, len(self.data_x_train) - self.window_size))
            examp.append(self.val[start_index[i]: start_index[i] + self.window_size])
        for i in range(self.test_num):
            for m, item in enumerate(examp[i]):
                if m < self.x_size:
                    self.date_pred[i, m, 0] = item[1]
                    self.date_pred[i, m, 1] = item[2]
                    self.pattern_pred[i, m, int(item[0])] = 1
                    self.user_pred[i, m, int(item[3] - 1)] = 1
                    self.item_pred[i, m, int(item[4] - 1)] = 1
                else:
                    self.pattern_truth[i, m - self.x_size] = int(item[0])

    def model_build(self):
        print('Model building......')

        pattern_input = Input(shape=(self.x_size, 5), dtype='float32', name='pattern_input')
        date_input = Input(shape=(self.x_size, 2), dtype='float32', name='date_input')
        user_input = Input(shape=(self.x_size, 5), dtype='float32', name='user_input')
        item_input = Input(shape=(self.x_size, 5), dtype='float32', name='item_input')

        user = Dense(64, input_shape=(self.x_size, 5), activation='sigmoid')(user_input)
        item = Dense(64, input_shape=(self.x_size, 5), activation='sigmoid')(item_input)
        x = Dense(64, activation='sigmoid')(pattern_input)
        d_y = Dense(64, activation='sigmoid')(date_input)
        p_y = merge([x, d_y], concat_axis=1)
        p_y = merge([p_y, user], concat_axis=1)
        p_y = merge([p_y, item], concat_axis=1)
        p_y = LSTM(64)(p_y)
        p_y = Dense(self.y_size * 5, activation='sigmoid', name='p_y')(p_y)
        p_y = Reshape((self.y_size, 5))(p_y)
        model = Model([pattern_input, date_input, user_input, item_input], p_y)
        optimizer = Nadam(lr=0.0018, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.fit([self.pattern_x, self.date_x, self.user_x, self.item_x], self.pattern_y, batch_size=128, nb_epoch=1)

        return model

    def model_predict(self, model):
        print('Model Predict......')
        predicts = model.predict([self.pattern_pred, self.date_pred, self.user_pred, self.item_pred], verbose=1)
        for predict in predicts[0]:
            ans = []
            for n, row in enumerate(predict):
                max1, max2 = [0, -1], [0, -1]
                for i, num in enumerate(row.tolist()):
                    if num >= max1[0]:
                        max2 = max1
                        max1 = [num, i]
                    elif num > max2[0]:
                        max2 = [num, i]
                ans.append([max1[1], max2[1]])

            # print(predicts)
            print("=====predict====")
            print(ans)
            print(self.pattern_truth)
        return predicts


def main():
    m = ModelBuilding()
    model = m.model_build()
    predict = m.model_predict(model)

if __name__ == '__main__':
    main()
