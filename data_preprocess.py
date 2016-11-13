# -*- coding: utf-8 -*-
from numpy import *

#workclass_value = [" Private", " Self-emp-not-inc", " Self-emp-inc", " Federal-gov", " Local-gov", " State-gov", " Without-pay", " Never-worked"]
workclass_value = [" Self-emp-inc", " Federal-gov", " Local-gov", " Self-emp-not-inc", " State-gov", " Private", " ?", " Without-pay", " Never-worked"]
#marital_status_value = [" Married-civ-spouse", " Divorced", " Never-married", " Separated", " Widowed", " Married-spouse-absent", " Married-AF-spouse"]
marital_status_value = [" Married-civ-spouse", " Married-AF-spouse", " Divorced", " Widowed", " Married-spouse-absent", " Separated", " Never-married"]
#occupation_value = [" Tech-support", " Craft-repair", " Other-service", " Sales", " Exec-managerial", " Prof-specialty", " Handlers-cleaners", " Machine-op-inspct", " Adm-clerical", " Farming-fishing", " Transport-moving", " Priv-house-serv", " Protective-serv", " Armed-Forces"]
occupation_value = [" Exec-managerial", " Prof-specialty", " Protective-serv", " Tech-support", " Sales", " Craft-repair", " Transport-moving", " Adm-clerical", " Machine-op-inspct", " Farming-fishing", " Armed-Forces", " ?", " Handlers-cleaners", " Other-service", " Priv-house-serv"]
#relationship_value = [" Wife", " Own-child", " Husband", " Not-in-family", " Other-relative", " Unmarried"]
relationship_value = [" Wife", " Husband", " Not-in-family", " Unmarried", " Other-relative", " Own-child"]
#race_value = [" White", " Asian-Pac-Islander", " Amer-Indian-Eskimo", " Other", " Black"]
race_value = [" Asian-Pac-Islander", " White", " Black", " Amer-Indian-Eskimo", " Other"]
#native_country_value = [" United-States", " Cambodia", " England", " Puerto-Rico", " Canada", " Germany", " Outlying-US(Guam-USVI-etc)", " India", " Japan", " Greece", " South", " China", " Cuba", " Iran", " Honduras", " Philippines", " Italy", " Poland", " Jamaica", " Vietnam", " Mexico", " Portugal", " Ireland", " France", " Dominican-Republic", " Laos", " Ecuador", " Taiwan", " Haiti", " Columbia", " Hungary", " Guatemala", " Nicaragua", " Scotland", " Thailand", " Yugoslavia", " El-Salvador", " Trinadad&Tobago", " Peru", " Hong", " Holand-Netherlands"]
native_country_value = [" Iran", " France", " India", " Taiwan", " Japan", " Yugoslavia", " Cambodia", " Italy", " England", " Canada", " Germany", " Philippines", " Hong", " Greece", " China", " Cuba", " ?", " Scotland", " United-States", " Hungary", " Ireland", " Poland", " South", " Thailand", " Ecuador", " Jamaica", " Laos", " Portugal", " Puerto-Rico", " Trinadad&Tobago", " Haiti", " El-Salvador", " Honduras", " Vietnam", " Peru", " Nicaragua", " Mexico", " Guatemala", " Columbia", " Dominican-Republic", " Outlying-US(Guam-USVI-etc)", " Holand-Netherlands"]

def string2code(value):
    dict = {}
    for i in range(len(value)):
        dict[value[i]] = i
    return dict

workclass_map = string2code(workclass_value)
marital_status_map = string2code(marital_status_value)
occupation_map = string2code(occupation_value)
relationship_map = string2code(relationship_value)
race_map = string2code(race_value)
native_country_map = string2code(native_country_value)

#缺失数据补齐
#workclass_map[' ?'] = workclass_map[" Private"]
#occupation_map[' ?'] = len(occupation_map)
#native_country_map[' ?'] = native_country_map[" United-States"]

def age2level(n):
    if n >= 61:
        return 5
    elif n <= 20:
        return 0
    else:
        return (n - 11) / 10

def education2level(n):
    if n < 13:
        return (n - 1) / 3
    else:
        return (n - 5) / 2

def country2level(n):
    if n < 24:
        return n / 8
    elif n < 33:
        return 3
    else:
        return 4

def row2dict(row):
    dict = []
    dict.append(age2level(int(row[0])))#age
    dict.append(workclass_map[row[1]])#workclass
    # 抽样权重不要dict['fnlwgt'] = long(row[2])
    # 与education-num重复dict['education'] = row[3]
    dict.append(education2level(int(row[4])))#education-num
    dict.append(marital_status_map[row[5]])
    dict.append(occupation_map[row[6]])
    dict.append(relationship_map[row[7]])
    dict.append(race_map[row[8]])
    dict.append(0 if row[9] == ' Male' else 1)
    dict.append(int(row[10]))#capital-gain
    dict.append(int(row[11]))#capital-loss
    dict.append(int(row[12]))#hours-per-week
    dict.append(country2level(native_country_map[row[13]]))
    # dict['income'] = 0 if row[14] == ' <=50K' else 1
    return dict