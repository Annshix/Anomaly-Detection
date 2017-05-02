# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import params


class DataProcessing:

    def __init__(self, written=0):
            self.df = pd.read_csv(params.file_path)
            self.feature = [[], []]
            self.written = written
            self.temp_data = self.__time_trans()

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
            self.feature[0].append(self.time_transfer(row[3]))
            self.feature[1].append(self.time_transfer(row[4]))
        ans = self.df.drop(self.df.columns[[3, 4, 5]], axis=1)\
            .assign(date=pd.Series(self.feature[0]))\
            .assign(duedate=pd.Series(self.feature[1]))
        print("Wub,End transferring time...")
        return ans

    @staticmethod
    def pair_generate(d):
        dt = d.sort_values(by='date').values
        pattern_pair = {}
        data_fin = []
        pair_num = 0
        for row in dt:
            user = row[2]
            item = row[3]
            pattern_pair.setdefault((user, item), [])
            pattern_pair[(user, item)].append(row)
        print("Lub, Begin generating pairs...")
        for key, values in pattern_pair.items():
            l = len(values)
            if l == 1:
                # behave = values[0]
                # duration = 0
                # print([behave[0], behave[0]])
                # event = [behave[0], behave[0]] + behave[2:5] + [duration]
                # data_fin.append(event)
                continue
            else:
                i, j = 0, 1
                while j < l:
                    behave1 = values[i]
                    behave2 = values[j]
                    duration = behave2[4] - behave1[4] + 0.5
                    event = [behave2[0], behave1[0], duration, behave2[4], behave1[6], behave1[7]]
                    data_fin.append(event)
                    pair_num += 1
                    i += 1
                    j += 1
        print(pair_num)
        return data_fin

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

    @staticmethod
    def item_classify(item_list, index):
        user_boundary = [100, 1000, 2000, 5000]
        item_boundary = [10, 100, 300, 1000]
        if index == 'user':
            boundary = user_boundary
        else:
            boundary = item_boundary
        map_set = dict()
        ans = []
        for item in item_list:
            map_set.setdefault(item, 0)
            map_set[item] += 1
        for item in item_list:
            if map_set[item] < boundary[0]:
                ans.append(1)
            elif map_set[item] < boundary[1]:
                ans.append(2)
            elif map_set[item] < boundary[2]:
                ans.append(3)
            elif map_set[item] < boundary[3]:
                ans.append(4)
            else:
                ans.append(5)
        return ans

    def clean_data(self):
        c = self.temp_data.T.values[2]
        i = self.temp_data.T.values[3]
        # print('transfer to index...')
        # # card_dict = self.name_to_index(c)
        # # item_dict = self.name_to_index(i)
        # print('end index transfer...')
        # card_num = list(map(lambda x: card_dict[x], c))
        # item_code = list(map(lambda x: item_dict[x], i))
        print('LubLub, start claasifying......')
        user_level = self.item_classify(c, 'user')
        print('user classified......')
        item_level = self.item_classify(i, 'item')
        print("end classifying...")

        temp = self.temp_data.assign(user_level=pd.Series(user_level))\
            .assign(item_level=pd.Series(item_level))
        paired_data = self.pair_generate(temp)
        pd.DataFrame(np.array(paired_data), columns=['pattern1', 'pattern2', 'duration', 'date',
                                                     'user_level', 'item_level'])\
            .to_csv('pair.csv')
        return


def main():
    m = DataProcessing()
    m.clean_data()

if __name__ == '__main__':
    main()

