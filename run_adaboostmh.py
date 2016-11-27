# -*- coding: utf-8 -*-
# adaboost MH
from adaboostmh import *
from sklearn import metrics

import time
from data_preprocess import loadData

classes_name = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']

class_num = len(classes_name)
# 数据读取
train_x, train_y, test_x, test_y = loadData('segmentation.data', 'segmentation.test')

# AdaboostMH
start = time.clock()
train_y_mat = mat(zeros((len(train_y), class_num)))

for j in xrange(len(train_y)):
    for i in xrange(class_num):
        if train_y[j] == classes_name[i]:
            train_y_mat[j, i] = 1
        else:
            train_y_mat[j, i] = -1

model = AdaBoostMH(class_num=7, max_iter=50)  # 基于决策树的adaboost，最大迭代次数max_iter
model.fit(train_x, train_y_mat)

end = time.clock()
print "learnning time: %f s" % (end - start)

# 预测
predictlist = []
for i in range(len(classes_name)):
    predict = model.predict(test_x, i)
    predictlist.append(predict)

predict_sum = []
for i in range(len(test_x)):
    l = [x[i] for x in predictlist]
    predict_sum.append(classes_name[l.index(max(l))])
print(metrics.classification_report(test_y, predict_sum))

print "ok"
