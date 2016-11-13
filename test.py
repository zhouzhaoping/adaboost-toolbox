# -*- coding: utf-8 -*-
from adaboost import *
from sklearn import metrics
import csv
import time
from data_preprocess import row2dict

#加载数据
reader = csv.reader(open('adult.data', 'r'))
train_x = []
train_y = []
for row in reader:
    #去除训练集中的未知项
    #if ' ?' not in row:
    train_x.append(row2dict(row))
    train_y.append(-1 if row[14] == ' <=50K' else 1)
print 'load %d train_data complete!' % (len(train_x))

#加载测试集
reader = csv.reader(open('adult.test', 'r'))
test_x = []
test_y = []
for row in reader:
    test_x.append(row2dict(row))
    test_y.append(-1 if row[14] == ' <=50K.' else 1)
print 'load %d test_data complete!' % (len(test_x))

#算法执行
start = time.clock()
model = AdaBoost(max_iter=50) # 基于决策树的adaboost，最大迭代次数max_iter
model.fit(train_x, train_y)
end = time.clock()
print "running time: %f s" % (end - start)

#预测
predict = model.predict(test_x)
print(metrics.classification_report(test_y, predict))