# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:19:55 2019

@author: xubing
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 20:40:34 2019

@author: xubing

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split as sp
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,classification_report
from sklearn.metrics import precision_recall_curve,roc_curve,roc_auc_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

file = 'data/processed.csv'
#file = 'data/BreastTissue.csv'
#file = 'data/glass.data'
num_leaves = 20
num_classes = 2

class nj_gbdtsoftmax():
    def __init__(self):
        self.name = 'gs'
        self.file = file
    
    def dataReader(file,sep = ',',header = 0,encoding = 'utf8'):
        '''
        读数据
        '''
        df = pd.read_csv(file,sep = sep,header = header,encoding = encoding )
        print('data shape:',df.shape)
        return df
    
    def dropCols(df,cols):
        '''
        删除列
        '''
        df.drop(cols,axis = 1,inplace = True)
        print('data shape:',df.shape)
        return df
    
    def nanFill(df,col,params):
        '''
        列 缺失值填充
        '''
        if params == 0:
            df = df[col].fillna(0)
        elif params == 'mean':
            df = df[col].fillna(df[col].mean())
        elif params == 'median':
            df = df[col].fillna(df[col].median())
        elif params == 'mode':
            df = df[col].fillna(df[col].mode)
        else:
            print('目前只能选择0、均、中、众值填充！')
        print('data info:',df.info())
        return df
    
    def dataSplit(self,df,label_col,test_size,Standardization = False,Normalization = False,Feature_selection = False):
        '''
        数据分割
        '''
        #string的类别映射到id
        classes = df[label_col].value_counts().index.tolist()
        class2id_map = {}
        for i in range(len(classes)):
            class2id_map[classes[i]] = i
        df[label_col].replace(class2id_map,inplace = True)
        print('map dict as follow:')#映射字典
        print('===========')
        for k,v in class2id_map.items():
            print('|',k,':',v,'|')
        print('==========')
        
        X = df.drop(label_col,axis = 1)
        y = df[label_col]
        
        if Standardization:
            self.dataStandardization(X)
            
        if Normalization:
            self.dataNormalization(X)
        if Feature_selection:
            self.featureSelection(X,y)
            
        
        
        X_train,X_test,y_train,y_test = sp(X,y,test_size = test_size,random_state = 2019)
        return X,X_train,X_test,y_train,y_test
    def featureSelection(k,X,y):
        '''特征筛选,卡方选择最优的K个特征'''
        selector = SelectKBest(score_func= chi2,k=k)
        selector.fit(X,y)
        print('every features score:',selector.scores_)
        X = selector.transform(X)
        
        return X
    
    def dataStandardization(X):
        '''数据标准化'''
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        return X
    def dataNormalization(X):
        '''数据正则化'''
        normalizer = Normalizer(norm = 'l2')
        X = normalizer.transform(X)
        return X
        
    
    def gbdtAlgorithm(X_train,X_test,y_train,y_test):
        lgb_train = lgb.Dataset(X_train,y_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',#'multiclass' 设置 num_class
            'metric': {'binary_logloss'},#
            'num_leaves': 20,
            'num_trees': 20,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
            'num_class':1#非多分类时，此参数应该设置为1，多分类时应该设置为类别个数
            
        } #二分类参数
#        params = {
#            'task': 'train',
#            'boosting_type': 'gbdt',
#            'objective': 'multiclass',#'multiclass' 设置 num_class
#            'metric': {'softmax'},#
#            'max_depth':5,
#            'num_leaves': num_leaves,
#            'num_trees': 10,
#            'learning_rate': 0.01,
#            'feature_fraction': 0.9,
#            'bagging_fraction': 0.8,
#            'bagging_freq': 5,
#            'verbose': 0,
#            'num_class':6#非多分类时，此参数应该设置为1，多分类时应该设置为类别个数
#            }#多分类参数
        
        
        # number of leaves,will be used in feature transformation
        
        print('Start training...')
        # train
        gbm = lgb.train(params=params,
                        train_set=lgb_train,
                        valid_sets=lgb_train,
                        )
        
        
        #直接使用gbdt的分类结果
        y_predict = gbm.predict(X_test)#二分类时只有一个概率值，多分类时有多个概率值，与上面的num_classes 相对应
        y_score = y_predict
        
        #二分类预测结果
        y_predict = [1 if (x>0.5) else 0  for x in y_predict]
        
        #多分类时预测结果
#        y_predict = [list(x).index(max(x)) for x in y_predict]
        
#        precision,recall,fscore,support  = precision_recall_fscore_support(y_test,result2)
#        print('\n精确率：%.4f\n召回率：%.4f\nf分数：%.4f\n支持数：%d'%(precision[0],recall[0],fscore[0],support[0]))
#        return precision,recall,fscore,support
        return gbm,y_test,y_predict,y_score
      
    
    def one_hot(target,one_hot_dimension):
        one_hot_target = np.array([[int(i == int(target[j])) for i in range(one_hot_dimension)] for j in range(len(target))])
        return list(one_hot_target.reshape(1,-1))[0]
    
    def transformData(gbm,X_train,X_test,y_train,y_test):
        '''
        数据经gbdt进行转换
        '''
        # y_pred分别落在n棵树上的哪个叶子节点上
        y_pred = gbm.predict(X_train, pred_leaf=True)
        #二分类只显示关注的类别落在哪个叶子节点上，
        #多分类时n个类别分别落在哪个叶子节点上
        
        
        
#        array([[4, 0, 2, 2, 4, 1], #多分类时，这里应该拉成一个长向量
#       [3, 3, 4, 2, 3, 2],
#       [1, 3, 2, 2, 4, 4],
#       [1, 3, 0, 2, 3, 3],
#       [4, 3, 0, 2, 2, 5],
#       [3, 3, 5, 5, 2, 4],
#       [3, 3, 4, 2, 3, 3],
#       [4, 3, 3, 2, 4, 3],
#       [4, 3, 4, 2, 3, 3],
#       [3, 3, 4, 2, 4, 3]])
        
        print('Writing transformed training data')
        transformed_training_matrix = np.zeros([len(y_pred),len(y_pred[0])*num_leaves])
        for i in range(len(y_pred)):
            transformed_training_matrix[i] = nj.one_hot(y_pred[i],num_leaves)
        
        y_pred = gbm.predict(X_test, pred_leaf=True)
        print('Writing transformed testing data')
        transformed_testing_matrix = np.zeros([len(y_pred),len(y_pred[0])*num_leaves])
        for i in range(len(y_pred)):
            transformed_testing_matrix[i] = nj.one_hot(y_pred[i],num_leaves)  
        return y_pred,transformed_training_matrix,transformed_testing_matrix
    
    def lrAlgorithm(X_train,X_test,y_train,y_test):
        lr = LogisticRegression(penalty='l1')
        lr.fit(X_train,y_train)
        y_predict = lr.predict(X_test)
        y_score = lr.predict_proba(X_test)
        
#        print('------LR----')
#        precision,recall,fscore,support  = precision_recall_fscore_support(y_test,y_)
#        print('\n精确率：%.4f\n召回率：%.4f\nf分数：%.4f\n支持数：%d'%(precision[0],recall[0],fscore[0],support[0]))        
#        return precision,recall,fscore,support
        return y_test,y_predict,y_score
    def evalution(y_test,y_predict):
        conf_mat = confusion_matrix(y_test,y_predict)
        class_report = classification_report(y_test,y_predict)
        print('混淆矩阵:',conf_mat)
        print('评估指标:',class_report)
        
        #return conf_mat,class_report
    def overfitAvoid():
        '''调参,防止过拟合'''
        return 0
    def accuaryLift():
        '''调参,提高准确率'''
        return 0
    
    def roc_curve_Plot(y_test,y_score,label_col):
        '''roc曲线绘制'''
        one_hot_encode = np.array(pd.get_dummies(pd.DataFrame(y_test)[label_col]))
        y_test = one_hot_encode
#        print(y_test)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
            #print(fpr[i],tpr[i])
            roc_auc[i] = roc_auc_score(y_test[:,i],y_score[:,i])
            
            ax.plot(fpr[i],tpr[i],label = 'target = %s,auc = %s'%(i,roc_auc[i]))
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC_curve')
        ax.legend(loc = 'best')
        ax.set_xlim(0,1.0)
        ax.set_ylim(0,1.0)
        ax.grid()
        plt.show()
        

    def pr_curve_Plot(y_test,y_score,label_col):
        '''pr曲线绘制'''
                #多分类
        one_hot_encode = np.array(pd.get_dummies(pd.DataFrame(y_test)[label_col]))
        y_test = one_hot_encode
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        precison = dict()
        recall = dict()
        for i in range(num_classes-3):
            precison[i],recall[i],_ = precision_recall_curve(y_test[:,i],y_score[:,i])
            ax.plot(recall[i],precison[i],label = 'target = %s'%i)
        ax.set_xlabel('Recall_score')
        ax.set_ylabel('Precision_score')
        ax.set_title('P-R_curve')
        #ax.legend(loc = 'best')
        ax.set_xlim(0,1.0)
        ax.set_ylim(0,1.0)
        ax.grid()
        plt.show()
    #应该还有两个函数，一个是特征筛选，一个是数据的标准化
        
        
    
if __name__ == '__main__':
    nj = nj_gbdtsoftmax    
    
    #二分类模型
    df = nj.dataReader(file)
    df = nj.dropCols(df,['SUBS_INSTANCE_ID','PRODUCT_TYPE'])
    df = nj.nanFill(df,df.columns,0)
    X,X_train,X_test,y_train,y_test = nj.dataSplit(nj,df,'IS_LOST',0.2)
    gbm,y_test,y_predict,y_predict_prob= nj.gbdtAlgorithm(X_train,X_test,y_train,y_test)
    print('----只使用GBDT的模型效果----')
    nj.evalution(y_test,y_predict)
    

    
    print('----只使用LR的模型效果-------')
    y_test,y_predict,y_score = nj.lrAlgorithm(X_train,X_test,y_train,y_test)
    nj.evalution(y_test,y_predict)
    nj.roc_curve_Plot(y_test,y_score,'IS_LOST')
    nj.pr_curve_Plot(y_test,y_score,'IS_LOST')
    
    print('----使用GBDT+LR的模型效果----')
    y_pred,X_train,X_test = nj.transformData(gbm,X_train,X_test,y_train,y_test)
    y_test,y_predict,y_score = nj.lrAlgorithm(X_train,X_test,y_train,y_test)
    nj.evalution(y_test,y_predict)
    nj.roc_curve_Plot(y_test,y_score,'IS_LOST')
    nj.pr_curve_Plot(y_test,y_score,'IS_LOST')
    
    
    
#    #多分类模型   
##    df = nj.dataReader(file,sep = '\t')
##    X_train,X_test,y_train,y_test = nj.dataSplit(df,'Class',0.2)#数据没有经过标准化
#    
#    
#    df = nj.dataReader(file,header = None)
#    df = nj.dropCols(df,0)
#    X,X_train,X_test,y_train,y_test = nj.dataSplit(nj,df,10,0.4,Standardization=True)
#    
#    print('---------Only GBDT-------------')#只使用GBDT模型效果
#    gbm,y_test,y_predict,y_score = nj.gbdtAlgorithm(X_train,X_test,y_train,y_test)
#    nj.evalution(y_test,y_predict)
#    nj.roc_curve_Plot(y_test,y_score,10)
#    nj.pr_curve_Plot(y_test,y_score,10)
#    
#    print('---------Only Softmax-----------')#只使用Softmax模型效果
#    y_test,y_predict,y_score = nj.lrAlgorithm(X_train,X_test,y_train,y_test)
#    nj.evalution(y_test,y_predict)
#    nj.roc_curve_Plot(y_test,y_score,10)
#    nj.pr_curve_Plot(y_test,y_score,10)
#    
#    
#    
#    print('--------GBDT+Softmax-----------')#GBDT做特征处理，Softmax用来分类
#    y_pred,mat1,mat2 = nj.transformData(gbm,X_train,X_test,y_train,y_test)
#    y_test,y_predict,y_score = nj.lrAlgorithm(mat1,mat2,y_train,y_test)
#    nj.evalution(y_test,y_predict)
#    print(y_pred[10].reshape(-1,num_classes))
#    nj.roc_curve_Plot(y_test,y_score,10)
#    nj.pr_curve_Plot(y_test,y_score,10)

    
    
    
