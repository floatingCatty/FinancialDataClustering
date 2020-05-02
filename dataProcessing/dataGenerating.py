import os
import pandas as pd
import numpy as np
import dataProcessing.utils as utils
from tqdm import tqdm
import pymysql

orgPath = "E:\Mine\education\\University\contest\\fuwu\\trainingSetM\Data_FCDS_hashed"
entPath = "E:\Mine\education\\University\contest\\fuwu\\trainingSetM\entname.txt"

class dataProcess(object):
    def __init__(self, ):
        self.connection = pymysql.connect(host='39.106.110.16',
                             user='liyifan',
                             password='Liyifan123',
                             db='fuchuang',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)



def processing():
    nameList = utils.getNameList(orgPath)
    nameList.remove('company_baseinfo.csv')
    count = 0
    main_table = pd.read_csv(orgPath + "\\" + "company_baseinfo.csv", encoding='ISO-8859-1', low_memory=False)
    # for file in tqdm(nameList):
    #     main_table = pd.merge(
    #         left=main_table,
    #         right=pd.read_csv(orgPath + "\\" + file, encoding='ISO-8859-1', low_memory=False),
    #         left_on="entname",
    #         right_on='entname',
    #         how='left'
    #     )
    main_table.to_csv(orgPath+"\\"+"main_table.csv", encoding='ISO-8859-1')

    return True

if __name__ == '__main__':
    processing()


