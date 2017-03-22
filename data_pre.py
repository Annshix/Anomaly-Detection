# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import params


class DataProcessing:

    def __init__(self, written):
        self.df = pd.read_csv(params.file_path)
        self.feature = [[], []]
        self.written = written
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
        if self.written:
            return
        for row in self.df.values:
            ft = map(lambda x: self.time_transfer(x), row[3:4])
            map(lambda x: self.feature[x].append(ft[x]), [0, 1])
        ans = self.df.drop(self.df.columns[[3, 4, 5]], axis=1)\
            .assign(date=pd.Series(self.feature[0]))\
            .assign(duedate=pd.Series(self.feature[1]))
        ans.to_csv(params.re_file_path)

    @staticmethod
    def name_to_index(obj):
        prev = -1
        index = 0
        map_set = dict()
        for item in obj:
            if prev != item:
                index += 1
            map_set[item] = index
            prev = item
        return map_set

    def clean_data(self):
        dt = pd.read_csv(params.re_file_path)
        c = dt.sort_values(by='CardNum' and 'ItemCode').T.values[3]
        i = dt.sort_values(by='CardNum' and 'ItemCode').T.values[4]
        card_dict = self.name_to_index(c)
        item_dict = self.name_to_index(i)
        card_num = list(map(lambda x: card_dict[x], c))
        item_code = list(map(lambda x: item_dict[x], i))

        ans = dt.assign(card_num=pd.Series(card_num))\
            .assign(item_code=pd.Series(item_code))\
            .drop(dt.columns[[2, 3]], axis=1)

        return ans


def main():
    m = DataProcessing()
    m.clean_data()


if __name__ == '__main__':
    main()

