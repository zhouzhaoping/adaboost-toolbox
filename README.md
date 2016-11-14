[Image Segmentation data](http://archive.ics.uci.edu/ml/datasets/Image+Segmentation)   
####基础Adaboost二分类
- decision_stump.py:  
单层决策树的实现，buildStump接口找到当前加权数据集上的最优单层决策树
- adaboost.py:  
基于单层决策树adaboost算法的类Adaboost，关键方法fit学习训练集和predict预测
- data_preprocess.py:  
数据预处理

####OVR(*one-versus-rest*)
- one_versus_rest.py:  
调用Adaboost使用1vN算法实现多分类

####OVO
*one-versus-one*

####Adaboost.MH等算法
*Improved boosting algorithms using confidence-rated predictions*


