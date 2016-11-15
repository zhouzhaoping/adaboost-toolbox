# -*- coding: utf-8 -*-
# 1vN算法实现多分类
from adaboost import *
from sklearn import metrics

import time
from data_preprocess import loadData

classes_name = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']

# 数据读取
train_x, train_y, test_x, test_y = loadData('segmentation.data', 'segmentation.test')

# 1vN算法执行
start = time.clock()
modellist = []
for i in range(len(classes_name)):
    model = AdaBoost(max_iter=50) # 基于决策树的adaboost，最大迭代次数max_iter
    new_train_y = []
    for x in train_y:
        if x == classes_name[i]:
            new_train_y.append(1)
        else:
            new_train_y.append(-1)
    model.fit(train_x, new_train_y)
    modellist.append(model)
end = time.clock()
print "learnning time: %f s" % (end - start)

# 预测
predictlist = []
for i in range(len(classes_name)):
    predict = modellist[i].predict(test_x)
    predictlist.append(predict)
predict_sum = []
for i in range(len(test_x)):
    l = [x[i] for x in predictlist]
    predict_sum.append(classes_name[l.index(max(l))])
print metrics.classification_report(test_y, predict_sum)
