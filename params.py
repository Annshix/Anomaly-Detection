# -*- coding: utf-8 -*-

driver = '{SQL Server}'
server = '10.58.82.94'
db = 'YFY_TW'
uid = 'sa'
pwd = 'SAPB1Admin'

file_path = 'data_raw.csv'
re_file_path = 'data.csv'

beh_num = 5
beh_patterns = [("ORDR", "RDR1"), ("ORDN", "RDN1"), ("OINV", "INV1"), ("ODLN", "DLN1"), ("OPDN","PDN1")]

headers = ["Pattern", "DocEntry", "CardCode", "DocDate", "DocDueDate", "TaxDate", "ItemCode"]

start_date = '2000-01-01'
end_date = '2020-01-01'
