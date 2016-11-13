# -*- coding: utf-8 -*-
# 单层决策树
from numpy import *

# 通过阈值比较对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

# 遍历stumpClassify函数所有的可能输入值，并找到数据集上最佳的单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # 初始化最小的错误和为无穷大
    for i in range(n):  # 遍历所有的特征
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps # 自适应之后阈值变化的步长
        for j in range(-1, int(numSteps) + 1):  # 遍历特征的切分点
            for inequal in ['lt', 'gt']:  # 遍历小于或者大于两种比较
                threshVal = (rangeMin + float(j) * stepSize) # 当前阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 调用stump classify（特征、阈值、比较方式）
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # 计算加权错误率
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst
