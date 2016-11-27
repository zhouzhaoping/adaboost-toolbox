# -*- coding: utf-8 -*-
# 1v1算法实现多分类
from adaboost import *
from sklearn import metrics

import time
from data_preprocess import loadData

classes_name = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']

# 数据读取
train_x, train_y, test_x, test_y = loadData('segmentation.data', 'segmentation.test')
#test_x, test_y, train_x, train_y = loadData('segmentation.data', 'segmentation.test')
train_x_list = [[], [], [], [], [], [], []]
for i in range(len(train_x)):
    index = classes_name.index(train_y[i])
    train_x_list[index].append(train_x[i])

# 1v1算法执行
start = time.clock()
# 生成模型矩阵
modelmatrix = [None] * len(classes_name)
for i in range(len(modelmatrix)):
    modelmatrix[i] = [AdaBoost(max_iter=50)] * len(classes_name)  # 基于决策树的adaboost，最大迭代次数max_iter
for i in range(len(classes_name)):
    for j in range(len(classes_name)):
        if i >= j:
            continue
        else:
            new_train_x = []
            new_train_y = []
            new_train_x.extend(train_x_list[i])
            new_train_x.extend(train_x_list[j])
            new_train_y = [1] * len(train_x_list[i])
            new_train_y.extend([-1] * len(train_x_list[j]))
            modelmatrix[i][j].fit(new_train_x, new_train_y)
end = time.clock()
print "learnning time: %f s" % (end - start)

# 预测
predictlist = []
for k in range(len(test_x)):
    predict = [0] * len(classes_name)
    for i in range(len(classes_name)):
        for j in range(len(classes_name)):
            if i < j:
                if modelmatrix[i][j].predict([test_x[k]]) == 1:
                    predict[i] += 1
                else:
                    predict[j] += 1
    index = predict.index(max(predict))
    predictlist.append(classes_name[index])
print len(predictlist)
print len(test_y)
count = 0
for i in range(len(predictlist)):
    if predictlist[i] == test_y[i]:
        count += 1
print float(count) / len(test_y)

print metrics.classification_report(test_y, predictlist)
