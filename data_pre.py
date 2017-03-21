# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import params
import csv


class DataProcessing:

    def __init__(self):
        self.df = pd.read_csv(params.file_path)
        self.df.sort_values(by='CardCode')
        self.feature = [[], []]
        self.__time_trans()

    rng = pd.date_range(params.start_date, params.end_date, freq='D')
    time_to_f = pd.Series(np.random.randn(len(rng)), index=rng)
    max_date = time_to_f.max()
    min_date = time_to_f.min()

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
            ft = list(map(lambda x: self.time_transfer(x), row[3:5]))
            map(lambda x: self.feature[x].append(ft[x]), [0, 1, 2])

    def clean_data(self):
        self.df.assign(date=pd.Series(self.feature[0]))
        self.df.assign(duedate=pd.Series(self.feature[1]))
        self.df.assign(taxdate=pd.Series(self.feature[2]))
        self.df.drop(self.df.columns[[3, 4, 5, 6]], axis=1)
        self.df.to_csv(params.re_file_path)


def main():
    m = DataProcessing()


if __name__ == '__main__':
    main()

