# -*- coding: utf-8 -*-
from decision_stump import *
import numpy as np

class AdaBoostMR:
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

    def __init__(self, class_num ,max_iter=40):
        self.__max_iter = max_iter
        self.__class_num = class_num


    __weakClassArr = []
    def fit(self, X, y):
        self.__weakClassArr = []
        m = shape(X)[0]
        D = np.array(ones((m, self.__class_num,self.__class_num)))  # 初始化训练数据的权值分布
        #init D
        for i in range(m):
            for l0 in range(self.__class_num):
                for l1 in range(self.__class_num):
                    if y[i,l0] == -1 and y[i,l1] == 1:
                        tmp  =  (y[i,:].T == mat(ones((self.__class_num, 1))))
                        Yi = tmp.sum()
                        Yi_else = self.__class_num - Yi
                        D[i,l0,l1] = 1.0/(m*Yi*Yi_else)
                    else:
                        D[i, l0, l1] = 0

        print "D.sum(): "+str(D.sum())
        D = D / D.sum()


        aggClassEst = mat(zeros((m, self.__class_num))) # 类别估计累计值
        for i in range(self.__max_iter):
            # 基于单层决策树
            bestArgsList = []
            classEstList = []

            for ii in range(self.__class_num):
                bestArgs, error, classEst = buildStump(X, y[:,ii].flatten().tolist(), np.apply_over_axes(sum, D, [1,2]).ravel())  # 获得单层决策树的参数，分类误差率，该分类器的分类结果
                bestArgsList.append(bestArgs)
                classEstList.append(classEst)

            self.__weakClassArr.append(bestArgsList)  # 存储参数
            # print "classEst: ",classEst.T

            alpha = 0
            for dat_i in range(m):
                for class_l0 in range(self.__class_num):
                    for class_l1 in range(self.__class_num):
                        alpha = alpha + D[dat_i, class_l0, class_l1] * (
                        self.single_weak_classifier(X[dat_i], i, class_l1) - self.single_weak_classifier(X[dat_i], i, class_l0))

            alpha = 0.5 * alpha
            alpha = 0.5 * exp((1 + alpha) / max((1 - alpha), 1e-16))

            print "alpha: %.4f"%(alpha)

            # 更新权值

            for (ii, ii_value) in enumerate(D):
                for (l0, l0_value) in enumerate(ii_value):
                    for (l1, l1_value) in enumerate(l0_value):
                        D[i+1,l0,l1] = D[i,l0,l1]*exp(0.5*alpha*(self.single_weak_classifier(X[ii],i,l0)-self.single_weak_classifier(X[ii],i,l1)))


            D = D / D.sum()
            # 更新类别估计累计值
            aggClassEst += alpha * classEst
            # print "aggClassEst: ",aggClassEst.T
            aggErrors = multiply(sign(aggClassEst) != mat(y).T, ones((m, 1)))
            errorRate = aggErrors.sum() / m
            print "total error: ", errorRate
            if errorRate == 0.0: break # 没有错误提前退出

        return self.__weakClassArr, aggClassEst

    def single_weak_classifier(self, X, classifier_id, label_id):
        dataMatrix = mat(X)
        m = shape(dataMatrix)[0]
        return  stumpClassify(dataMatrix, self.__weakClassArr[classifier_id][label_id]['dim'], self.__weakClassArr[classifier_id][label_id]['thresh'], self.__weakClassArr[classifier_id][label_id]['ineq'])

    def predict(self, X,class_ind):
        dataMatrix = mat(X)
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zeros((m, 1)))
        for i in range(len(self.__weakClassArr)):
            classEst = stumpClassify(dataMatrix, self.__weakClassArr[i][class_ind]['dim'], self.__weakClassArr[i][class_ind]['thresh'], self.__weakClassArr[i][class_ind]['ineq'])
            aggClassEst += self.__weakClassArr[i][class_ind]['alpha'] * classEst
            # print aggClassEst
        return sign(aggClassEst)