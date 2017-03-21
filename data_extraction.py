# -*- coding: utf-8 -*-

import pyodbc
import csv
import time
import sql_rel

import params


class DataExtraction:
    def __init__(self, written=False):
        self.beh_patterns = params.beh_patterns
        self.written = written
        self.beh_num = params.beh_num

    @staticmethod
    def conn_db():
        db = pyodbc.connect(
            'DRIVER=' + params.driver + ';SERVER=' + params.server +
            ';DATABASE=' + params.db + ';UID=' + params.uid + ';PWD=' + params.pwd)
        cursor = db.cursor()
        return cursor

    def generate(self):
        if self.written:
            return
        cursor = self.conn_db()

        f = open(params.file_path, 'w', encoding='utf-8', errors='ignore')
        f_csv = csv.writer(f)
        f_csv.writerow(params.headers)

        for i in range(self.beh_num):
                pattern = self.beh_patterns[i]
                cursor.execute(sql_rel.sql % (i, pattern[0], pattern[1]))
                for row in cursor.fetchall():
                    map(lambda x: time.mktime(row[x].timetuple()), [3, 4, 5])
                    f_csv.writerow(row)
        f.close()

        return


def main():
    m = DataExtraction()
    m.generate()

if __name__ == '__main__':
    main()





