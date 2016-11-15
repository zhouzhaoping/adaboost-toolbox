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
<pre>
             precision    recall  f1-score   support

  BRICKFACE       0.55      0.99      0.71       300
     CEMENT       0.83      0.65      0.73       300
    FOLIAGE       0.88      0.81      0.84       300
      GRASS       0.99      0.98      0.99       300
       PATH       1.00      0.91      0.95       300
        SKY       0.99      1.00      1.00       300
     WINDOW       0.86      0.51      0.64       300

avg / total       0.87      0.84      0.84      2100
</pre>
####OVO(*one-versus-one*)
- one_versus_rest.py: 
调用Adaboost使用1v1算法实现多分类
<pre>
             precision    recall  f1-score   support

  BRICKFACE       0.19      1.00      0.32       300
     CEMENT       0.00      0.00      0.00       300
    FOLIAGE       0.64      0.31      0.41       300
      GRASS       1.00      0.95      0.97       300
       PATH       0.00      0.00      0.00       300
        SKY       0.03      0.00      0.01       300
     WINDOW       0.87      0.11      0.20       300

avg / total       0.39      0.34      0.27      2100
</pre>
####Adaboost.MH等算法
*Improved boosting algorithms using confidence-rated predictions*


