# -*- coding: utf-8 -*-
from decision_stump import *


class AdaBoost:
    '''
    weak_classifiers：
    已经实现的弱分类器，有如下选择：
    stump-单层决策树

    max_iter：大于零的正整数
    最大迭代次数，默认四十次

    weak_classifier：
    算法选择的弱分类器，只能在列表weak_classifiers中取

    weakClassArr：
    训练之后保存的需要分类的信息，包含多个字典，每个字典都保存弱分类的信息
    {dim，ineq，thresh，alpha}

    fit(self, X, y)：
    对训练集的学习，X为训练集的数据矩阵（每一行是一个数据，每一列表示一个特征），y为标记{1.0,-1.0}

    predict(self, X)：
    对测试集进行预测，输出是一个标记构成的列表

    printDetail(self)：
    测试用的函数
    '''

    weak_classifiers = ['stump']
    __max_iter = 40

    def __init__(self, max_iter=40):
        self.__max_iter = max_iter

    __weakClassArr = []
    def fit(self, X, y):
        self.__weakClassArr = []
        m = shape(X)[0]
        D = mat(ones((m, 1)) / m)  # 初始化训练数据的权值分布
        aggClassEst = mat(zeros((m, 1))) # 类别估计累计值
        for i in range(self.__max_iter):
            # 基于单层决策树
            bestArgs, error, classEst = buildStump(X, y, D)  # 获得单层决策树的参数，分类误差率，该分类器的分类结果
            # print "D:",D.T
            alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 计算alpha系数
            bestArgs['alpha'] = alpha
            self.__weakClassArr.append(bestArgs)  # 存储参数
            # print "classEst: ",classEst.T
            # 更新权值
            expon = multiply(-1 * alpha * mat(y).T, classEst)
            D = multiply(D, exp(expon))
            D = D / D.sum()
            # 更新类别估计累计值
            aggClassEst += alpha * classEst
            # print "aggClassEst: ",aggClassEst.T
            aggErrors = multiply(sign(aggClassEst) != mat(y).T, ones((m, 1)))
            errorRate = aggErrors.sum() / m
            print "total error: ", errorRate
            if errorRate == 0.0: break # 没有错误提前退出
        return self.__weakClassArr, aggClassEst

    def predict(self, X):
        dataMatrix = mat(X)
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zeros((m, 1)))
        for i in range(len(self.__weakClassArr)):
            classEst = stumpClassify(dataMatrix, self.__weakClassArr[i]['dim'], self.__weakClassArr[i]['thresh'], self.__weakClassArr[i]['ineq'])
            aggClassEst += self.__weakClassArr[i]['alpha'] * classEst
            # print aggClassEst
        return sign(aggClassEst)