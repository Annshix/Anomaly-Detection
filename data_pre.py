# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import params
import csv


class DataProcessing:

    rng = pd.date_range(params.start_date, params.end_date, freq='D')
    time_to_f = pd.Series(np.random.randn(len(rng)), index=rng)
    max_date = time_to_f.max()
    min_date = time_to_f.min()

    def __init__(self):
        self.df = pd.read_csv(params.file_path)
        self.df.sort_values(by='CardCode')
        self.feature = [[], []]
        self.__time_trans()

    def time_transfer(self, t):
        try:
            return self.time_to_f[t]
        except KeyError:
            if t > params.end_date:
                return self.max_date
            else:
                return self.min_date
            pass

    def __time_trans(self):
        for row in self.df.values:
            try:
                self.feature[0].append(time_to_f[row[3]])
                self.feature[1].append(time_to_f[row[4]])
            except KeyError:
                if row[4] > params.end_date:
                    self.feature[1].append(max_date)
                if row[3] < params.start_date:
                    self.feature[0].append(min_date)
                pass

    def clean_data(self):
        self.df.assign(date=pd.Series(self.feature[0]))
        self.df.assign(duedate=pd.Series(self.feature[1]))
        self.df.drop(self.df.columns[[3, 4, 5, 6]], axis=1)
        self.df.to_csv(params.re_file_path)


def main():
    m = DataProcessing()


if __name__ == '__main__':
    main()

