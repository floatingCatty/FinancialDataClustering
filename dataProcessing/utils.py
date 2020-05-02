# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import os




def getNameList(dirPath):
    for _,_,files in os.walk(dirPath):
        return files

def getEntnameList(Path):
    f = open(Path)
    s = f.read().split('\n')
    return s

def encoding(path):
    f = pd.read_csv(path, encoding='ISO-8859-1')
    print(f.shape[0])
    # print(f[f['entname']=='8f0c15a6c0d2c09a157feef19ddb8783'].to_numpy())

if __name__ == '__main__':
    # print(getEntnameList("E:\Mine\education\\University\contest\\fuwu\\trainingSetM\entname.txt"))
    encoding(
        path="E:\Mine\education\\University\contest\\fuwu\\trainingSetM\main_table.csv"
    )