# -*- coding: utf-8 -*-
from numpy import *
import csv
from itertools import islice

def row2dict(row):
    dict = []
    for i in range(1, 19):
        dict.append((float(row[i])))
    return dict

def loadData(trainfilename, testfilename):
    #加载数据
    reader = csv.reader(open(trainfilename, 'r'))
    train_x = []
    train_y = []
    for row in islice(reader, 5, None):
        train_x.append(row2dict(row))
        train_y.append(row[0])
    print 'load %d train_data complete!' % (len(train_x))
    #加载测试集
    reader = csv.reader(open(testfilename, 'r'))
    test_x = []
    test_y = []
    for row in islice(reader, 5, None):
        test_x.append(row2dict(row))
        test_y.append(row[0])
    print 'load %d test_data complete!' % (len(test_x))
    return train_x, train_y, test_x, test_y
