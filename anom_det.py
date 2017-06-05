# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:10:05 2017

@author: Jaya
"""


# coding: utf-8

# # Anomaly Detection Lab!
# Group 8: Azqa Nadeem, Jaya Chithra

get_ipython().magic(u'matplotlib inline')

import datetime
import time

import pandas as pd
from sklearn.metrics import mean_squared_error
from pandas import DataFrame, read_csv
from sklearn import svm
import statsmodels.api as api
import statsmodels.graphics.tsaplots as tsaplots
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from operator import itemgetter
from itertools import groupby
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from sklearn import decomposition
from matplotlib import cm as cm

def correlation_matrix(df, labels, filename):


    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 100)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    #labels=['FIT101','LIT101','MV101','P101','AIT201','AIT202','AIT203','FIT201','MV201','P203','P205','DPIT301','FIT301','LIT301','MV301','MV302','MV303',\
#'MV304','P301','P302','AIT401','AIT402','FIT401','LIT401','P402','UV401',\
#'AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','PIT501',\
#'PIT502','PIT503','FIT601','P602']
    ax1.minorticks_on()
    ax1.xaxis.set_ticks(range(0,len(labels)))
    ax1.set_xticklabels(labels, rotation=270)

    ax1.yaxis.set_ticks(range(0,len(labels)))
    ax1.set_yticklabels(labels)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1, -0.5, 0,0.5,.75,.8,.9,.95,1])
    plt.savefig(filename)
    plt.show()


def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%d/%m/%Y %H:%M:%S %p')
    return time.mktime(time_stamp)

def ParseData(filename):
    ah = open(filename, 'r')
    data = []#contains features
    #y = []#contains labels
    count = 0
    prevminute = 0
    fit101, lit101, mv101, p101, ait201, ait202, ait203, fit201 =0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    mv201, p203, p205, dpit301, fit301, lit301, mv301, mv302 = 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0
    mv303, mv304, p301,p302, ait401, ait402, fit401, lit401 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ait501, ait502, ait503, ait504, fit501, fit502, fit503, fit504, pit501, pit502, pit503 = 0.0 , 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    fit601, p602 = 0.0, 0.0
    label = 0
    ah.readline()#skip first line
    for line_ah in ah:

        timestamp = pd.to_datetime(line_ah.strip().split(',')[0],dayfirst=True)# date
        minute = timestamp.minute
        if count == 0:
            prevminute = timestamp.minute
        if minute == prevminute:
            count = count +1
            fit101 = fit101 + float(line_ah.strip().split(',')[1])
            '''
            lit101 = lit101 + float(line_ah.strip().split(',')[2])
            mv101 = mv101 + float(line_ah.strip().split(',')[3])
            p101 = p101 + float(line_ah.strip().split(',')[4] )
        
            ait201 = ait201+float(line_ah.strip().split(',')[6])
            ait202 = ait202+float(line_ah.strip().split(',')[7])
            ait203 = ait203+ float(line_ah.strip().split(',')[8])
            fit201 = fit201+float(line_ah.strip().split(',')[9])
            mv201 = mv201+float(line_ah.strip().split(',')[10])
            p203 = p203+float(line_ah.strip().split(',')[13])
            p205 = p205+float(line_ah.strip().split(',')[15])
   
            dpit301 = dpit301+float(line_ah.strip().split(',')[17])
            fit301 = fit301+float(line_ah.strip().split(',')[18])
            lit301 = lit301+float(line_ah.strip().split(',')[19])
            mv301 = mv301+float(line_ah.strip().split(',')[20])
            mv302 = mv302+float(line_ah.strip().split(',')[21])
            mv303 = mv303+float(line_ah.strip().split(',')[22])
            mv304 = mv304+float(line_ah.strip().split(',')[23])
            p301 = p301+float(line_ah.strip().split(',')[24])
            p302 = p302+float(line_ah.strip().split(',')[25])
        
            ait401 = ait401+float(line_ah.strip().split(',')[26])
            ait402 = ait402+float(line_ah.strip().split(',')[27])
            fit401 = fit401+float(line_ah.strip().split(',')[28])
            lit401 = lit401+float(line_ah.strip().split(',')[29])

            ait501 = ait501+float(line_ah.strip().split(',')[35])
            ait502 = ait502+float(line_ah.strip().split(',')[36])
            ait503 = ait503+float(line_ah.strip().split(',')[37])
            ait504 = ait504+float(line_ah.strip().split(',')[38])
            fit501 = fit501+float(line_ah.strip().split(',')[39])
            fit502 = fit502+float(line_ah.strip().split(',')[40])
            fit503 = fit503+float(line_ah.strip().split(',')[41])
            fit504 = fit504+float(line_ah.strip().split(',')[42])
            pit501 = pit501+float(line_ah.strip().split(',')[45])
            pit502 = pit502+float(line_ah.strip().split(',')[46])
            pit503 = pit503+float(line_ah.strip().split(',')[47])

            fit601 = fit601+float(line_ah.strip().split(',')[48])
            p602 = p602+float(line_ah.strip().split(',')[50])
            '''
            if line_ah.strip().split(',')[52] == 'Attack':
              label = 1
            else:
              label = 0#label fraud
            #label save
            
        
        else:
            data.append([fit101/count,timestamp,label])
            '''
            data.append([fit101/count, lit101/count, mv101/count, p101/count,
                     ait201/count, ait202/count, ait203/count, fit201/count, mv201/count, p203/count, p205/count, 
                     dpit301/count, fit301/count, lit301/count, mv301/count, mv302/count, mv303/count, mv304/count, p301/count, p302/count,
                     ait401/count, ait402/count, fit401/count, lit401/count,
                     ait501/count, ait502/count, ait503/count, ait504/count, fit501/count, fit502/count, fit503/count, fit504/count, pit501/count, pit502/count, pit503/count,
                     fit601/count, p602/count,timestamp, label])
            #print (data)
            '''
            #print(timestamp)
            count = 1
            prevminute = minute
             
        #if(minute)
        #timestamp = string_to_timestamp(line_ah.strip().split(',')[0])
            fit101 = float(line_ah.strip().split(',')[1])
            '''
            lit101 = float(line_ah.strip().split(',')[2])
            mv101 =  float(line_ah.strip().split(',')[3])
            p101 = float(line_ah.strip().split(',')[4] )
        
            ait201 =  float(line_ah.strip().split(',')[6])
            ait202 =  float(line_ah.strip().split(',')[7])
            ait203 =  float(line_ah.strip().split(',')[8])
            fit201 =  float(line_ah.strip().split(',')[9])
            mv201 =  float(line_ah.strip().split(',')[10])
            p203 =  float(line_ah.strip().split(',')[13])
            p205 =  float(line_ah.strip().split(',')[15])
   
            dpit301 =  float(line_ah.strip().split(',')[17])
            fit301 =  float(line_ah.strip().split(',')[18])
            lit301 =  float(line_ah.strip().split(',')[19])
            mv301 = float(line_ah.strip().split(',')[20])
            mv302 =  float(line_ah.strip().split(',')[21])
            mv303 =  float(line_ah.strip().split(',')[22])
            mv304 =  float(line_ah.strip().split(',')[23])
            p301 =  float(line_ah.strip().split(',')[24])
            p302 =  float(line_ah.strip().split(',')[25])
        
            ait401 =  float(line_ah.strip().split(',')[26])
            ait402 =  float(line_ah.strip().split(',')[27])
            fit401 =  float(line_ah.strip().split(',')[28])
            lit401 =  float(line_ah.strip().split(',')[29])

            ait501 =  float(line_ah.strip().split(',')[35])
            ait502 =  float(line_ah.strip().split(',')[36])
            ait503 =  float(line_ah.strip().split(',')[37])
            ait504 =  float(line_ah.strip().split(',')[38])
            fit501 =  float(line_ah.strip().split(',')[39])
            fit502 =  float(line_ah.strip().split(',')[40])
            fit503 =  float(line_ah.strip().split(',')[41])
            fit504 =  float(line_ah.strip().split(',')[42])
            pit501 =  float(line_ah.strip().split(',')[45])
            pit502 =  float(line_ah.strip().split(',')[46])
            pit503 =  float(line_ah.strip().split(',')[47])

            fit601 =  float(line_ah.strip().split(',')[48])
            p602 = float(line_ah.strip().split(',')[50])''' 
            if line_ah.strip().split(',')[52] == 'Attack':
              label = 1#label fraud
            else:
              label = 0#label save
             
            
        '''
        if line_ah.strip().split(',')[52] == 'Attack':
            label = 1#label fraud
        else:
            label = 0#label save
        
        
        #FEATURE SELECTION
        data.append([fit101, lit101, mv101, p101,
                     ait201, ait202, ait203, fit201, mv201, p203, p205, 
                     dpit301, fit301, lit301, mv301, mv302, mv303, mv304, p301, p302,
                     ait401, ait402, fit401, lit401,
                     ait501, ait502, ait503, ait504, fit501, fit502, fit503, fit504, pit501, pit502, pit503,
                     fit601, p602,timestamp,label]) 
        #print(data[]
        
    data.append([fit101/count, lit101/count, mv101/count, p101/count,
                     ait201/count, ait202/count, ait203/count, fit201/count, mv201/count, p203/count, p205/count, 
                     dpit301/count, fit301/count, lit301/count, mv301/count, mv302/count, mv303/count, mv304/count, p301/count, p302/count,
                     ait401/count, ait402/count, fit401/count, lit401/count,
                     ait501/count, ait502/count, ait503/count, ait504/count, fit501/count, fit502/count, fit503/count, fit504/count, pit501/count, pit502/count, pit503/count,
                     fit601/count, p602/count,timestamp, label])
    '''
    data.append([fit101/count,timestamp,label])
    return data

# Training the classifier and K-Fold cross-validation
def TrainClassifier(clf, sm, usx, usy, cutoff):
    total_frauds = 0
    non_frauds = 0
    TP, FP, FN, TN = 0, 0, 0, 0 #Per fold values
    aTP, aFP, aFN, aTN = 0, 0, 0, 0 #Average value
    y_real, y_proba, y_pred = [], [], []
    kf = KFold(n_splits=10, shuffle=True)
    #split the data into train set and test set
    for train_index, test_index in kf.split(usx):
        x_train, x_test = usx[train_index], usx[test_index]
        y_train, y_test = usy[train_index], usy[test_index]
        # SMOTE
        x_train, y_train = sm.fit_sample(x_train, y_train)
        #train the classifier
        clf.fit(x_train, y_train)
        predict_proba = clf.predict_proba(x_test)#the probability of each smple labelled to positive or negative
        preds = predict_proba[:,1]
        y_predict = (predict_proba[:,1]>cutoff).astype(int) #cutoff as input param
        #Evaluating the classifier
        for i in range(len(y_predict)):
            if y_test[i]==1:
                total_frauds = total_frauds+1
            if y_test[i] == 0:
                non_frauds = non_frauds+1
            if y_test[i]==1 and y_predict[i]==1:
                TP += 1
            if y_test[i]==0 and y_predict[i]==1:
                FP += 1
            if y_test[i]==1 and y_predict[i]==0:
                FN += 1
            if y_test[i]==0 and y_predict[i]==0:
                TN += 1
            # collect y_test, preds and y_predict of all fold
        y_real.append(y_test)
        y_proba.append(preds)
        y_pred.append(y_predict)

        # calculate average FP, TP, FN, TN
    aTP = TP/10
    aFP = FP/10
    aFN = FN/10
    aTN = TN/10
    print ('Average TP: '+ str(aTP))
    print ('Average FP: '+ str(aFP))
    print ('Average FN: '+ str(aFN))
    print ('Average TN: '+ str(aTN))
    print ('Average actual frauds: ' + str(total_frauds/10))
    print ('Average actual legits: ' + str(non_frauds/10))
    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)
    y_proba= np.concatenate(y_proba)
    return (y_real, y_pred, y_proba)

if __name__ == "__main__":
    #src = 'normalize_normal.csv'
    src = 'normalize_normal.csv'
    testFile = 'normalize_attack.csv'
    ####################### Correlation ################################
    #dd = read_csv(src)
    #print(pd.to_datetime(dd.Timestamp, dayfirst=True))
    '''
    p1 = {'FIT101': df.FIT101, 'LIT101': df.LIT101, 'MV101': df.MV101, 'P101': df.P101}
    p1_labels = ['FIT101', 'LIT101', 'MV101', 'P101']
    p2 = {'AIT201': df.AIT201, 'AIT202': df.AIT202, 'AIT203': df.AIT203, 'FIT201' :df.FIT201, 'MV201': df.MV201, 'P203' :df.P203, 'P205' : df.P205}
    p2_labels = ['AIT201', 'AIT202', 'AIT203', 'FIT201','MV201', 'P203', 'P205' ]
    p3 = {'DPIT301' : df.DPIT301, 'FIT301': df.FIT301, 'LIT301' : df.LIT301, 'MV301' : df.MV301, 'MV302' : df.MV302, 'MV303' : df.MV303, 'MV304' : df.MV304, 'P301' :df.P301, 'P302' : df.P302}
    p3_labels= ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302']
    p4 = {'AIT401' : df.AIT401, 'AIT402': df.AIT402, 'FIT401' : df.FIT401, 'LIT401': df.LIT401, 'P402' : df.P402, 'UV401': df.UV401}
    p4_labels = ['AIT401', 'AIT402', 'FIT401','LIT401', 'P402' ,'UV401']
    p5 = {'AIT501':df.AIT501, 'AIT502': df.AIT502, 'AIT503': df.AIT503, 'AIT504': df.AIT504, 'FIT501': df.FIT501, 'FIT502': df.FIT502, 'FIT503': df.FIT503, 'FIT504' : df.FIT504, 'P501': df.P501, 'PIT501': df.PIT501, 'PIT502': df.PIT502, 'PIT503': df.PIT503}
    p5_labels = ['AIT501', 'AIT502', 'AIT503' , 'AIT504', 'FIT501', 'FIT502', 'FIT503' ,'FIT504', 'P501', 'PIT501', 'PIT502', 'PIT503']
    p6 = {'FIT601':df.FIT601, 'P602': df.P602}
    p6_labels = ['FIT601', 'P602']
    '''
    #testdf = read_csv(testFile)
    '''
    correlation_matrix(DataFrame(data=p1), p1_labels, 'p1.png')
    correlation_matrix(DataFrame(data=p2), p2_labels, 'p2.png')
    correlation_matrix(DataFrame(data=p3), p3_labels, 'p3.png')
    correlation_matrix(DataFrame(data=p4), p4_labels, 'p4.png')
    correlation_matrix(DataFrame(data=p5), p5_labels, 'p5.png')
    correlation_matrix(DataFrame(data=p6), p6_labels, 'p6.png')

    features = p1
    features.update(p2)
    features.update(p3)
    features.update(p4)
    features.update(p5)
    features.update(p6)
    '''
    df = ParseData(src)
    testdf = ParseData(testFile)
    
    #############################################################

   
    '''
           
    #ARMA
    dat = DataFrame(data={'timestamp' : pd.to_datetime(df.Timestamp, dayfirst=True), 'FIT101' : df.FIT101})
    dat.set_index("timestamp", inplace=True)
    tsaplots.plot_acf(dat)
    plt.show()
    model = api.tsa.ARMA(dat, (2,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())
    '''
    
    X,timestamp = [],[]
    Xt, timestampt, testlabel = [] , [], []
    for item in df:
        #split data into x,y
        X.append(item[0])
        
        timestamp.append(item[1])
       
    for item in testdf:
        #split data into x,y
        Xt.append(item[0])
        
        timestampt.append(item[1])
        testlabel.append(item[2])
        
    dat = DataFrame(data={'timestamp' : np.array(timestamp), 'FIT101' : np.array(X)})
    dat.set_index("timestamp", inplace=True)
    tsaplots.plot_acf(dat)
    plt.show()
    testdat = DataFrame(data={'timestamp': np.array(timestampt),'FIT101' : np.array(Xt)})
    testdat.set_index("timestamp", inplace=True)
    tsaplots.plot_acf(testdat)
    plt.show()
    
    xtrain = dat.values
    xtest = testdat.values
    
    history = [x for x in xtrain]
    predictions = list()
    
    '''
    model = api.tsa.ARMA(dat, (2,0))
    model_fit = model.fit()
    model_test = api.tsa.ARMA(testdat, (2,0))
    model_test_fit = model_test.fit()
  
    print(model_fit.aic)
    prediction =  model_test_fit.forecast()[0]
    print (prediction)
    err = testdf.FIT101 - prediction.values
    DataFrame(err).plot()
    pred = ((err > 0.01) | (err<-0.01))
    '''
    xtest = np.array(xtest)
    xtestnew = []
    residtestarray = []
    TN, TP, FN, FP, totalattack, totalnormal, precision, recall, f1 = 0, 0, 0, 0, 0,0, 0, 0, 0
     
    '''
    model = api.tsa.ARMA(history, (2,0))
    model_fit = model.fit(disp=0)
    residual = model_fit.resid
    print('###########Residuals############################')
    DataFrame(residual).plot(title='train')
    maxres = max(residual)
    numres = np.mean(residual)+len(residual)*np.std(residual)
    print('##############training')
    print('numres', numres )
    print('std test',np.std(residual))
    print('mean test',np.mean(residual))
    '''
    indexarray = []
    for t in range(0,1000):
        model = api.tsa.ARMA(history, (2,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = xtest[t]
        
        #actual - predicted 
        residtest = obs-yhat
        residtestarray.append(residtest)
        
        #attack or normal
        if residtest > 0.75 or residtest < -0.75:
            indexarray.append(1)
            #print('caught')
        else:
            indexarray.append(0)
        
        history.append(obs)
        xtestnew.append(xtest[t])
    
    #evaluate    
    print (indexarray)
    for v in range (0,1000):
        if testlabel[v] == 1:
            totalattack +=1
        if testlabel[v] == 0:
            totalnormal += 1
        if indexarray[v]==1 and testlabel[v] == 1 :
            TP += 1
        elif indexarray[v]==1 and testlabel[v] == 0:
            FP += 1
        elif indexarray[v]==0 and testlabel[v] == 1:
            FN += 1
        elif indexarray[v]==0 and testlabel[v] == 0:
            TN += 1
    print('Total attack:', totalattack)
    print('Total normal:', totalnormal)
    print('TP:', TP)
    print('FP:', FP)
    print('TN:', TN)
    print('FN:', FN)
    precision = TP/(TP+FP)
    recall =TP/ (TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    print('Precision',precision)
    print('recall',recall)
    print ('f1:',f1)
    '''
    error = mean_squared_error(np.array(xtestnew), predictions)
    print('########### Test Residuals############################')
    DataFrame(resitest).plot(title='test')
    maxrest = max(resitest)
    print('Test MSE: %.3f' % error)
    # plot
    #plt.plot(xtest)
    #plt.plot(predictions, color='red')
    #plt.show()
    
    '''
    
    '''
    pca = decomposition.PCA(n_components=10) # n_components=3 for dim red
    #x_array = np.array(X)
    y_array = np.array(y)
    t_array = np.array(ttest)
    ytest_array = np.array(ytest)
    
    #fit training and testing data for two components
    xtrain = pca.fit(np.array(X))
    Xtestnew = xtrain.transform(np.array(Xtest))
    #x_train, x_test, y_train, y_test = train_test_split(X, y_array, test_size = 0.2)#test_size:
    #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    #clf = IsolationForest(n_estimators = 100)
    #clf.fit(x_train)
    #clf.fit(X)
    #normal = Xtestnew[:,0]**2 +Xtestnew[:,1]**2 +Xtestnew[:,2]**2 +Xtestnew[:,3]**2 +Xtestnew[:,4]**2 + Xtestnew[:,5]**2 
    #abnormal = Xtestnew[:,6]**2 +Xtestnew[:,7]**2 +Xtestnew[:,8]**2 +Xtestnew[:,9]**2 
    
    
    #DataFrame(residuals[:,12]).plot(kind='line')
    #DataFrame(Xtestnew[:,0]).plot(kind='line')
    #DataFrame(Xtestnew[:,9]).plot(kind='line')
    #plot the components
    
   
    plt.plot(t_array,Xtestnew[:,0], '-', markersize=7, color='blue', alpha=0.5, label='class1')
    #plt.plot(X[20:40,0], t, '^', markersize=7, color='red', alpha=0.5, label='class2')
    
     
    
    plt.ylabel('Time')
    plt.xlabel('Component 1')
    
    #plt.xlim([-4,4])
    #plt.ylim([-4,4])
    #plt.grid()
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

    plt.show()
    
    plt.plot(t_array.year,Xtestnew[:,9], '-', markersize=7, color='blue', alpha=0.5, label='class1')
    #plt.plot(X[20:40,10], X[20:40,11], 'o', markersize=7, color='red', alpha=0.5, label='class2')
    plt.ylabel('Time')
    plt.xlabel('Component 10')
    
    #plt.xlim([-4,4])
    #plt.ylim([-4,4])
    #plt.grid()
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

    plt.show()
   '''
    '''
   # y_predict = clf.predict(x_test)
    y_predict = clf.predict(Xtest)
    print (len(y_predict))
    print (len(ytest_array))
    for i in range(len(y_predict)):
        if ytest_array[i]==1:
            total_anomalies = total_anomalies+1
        if ytest[i] == 0:
            total_normal = total_normal+1
        if ytest[i]==0 and y_predict[i]==1:
            TP += 1
        if ytest[i]==1 and y_predict[i]==1:
            FP += 1
        if ytest[i]==0 and y_predict[i]==-1:
            FN += 1
        if ytest[i]==1 and y_predict[i]==-1:
            TN += 1
    print (total_anomalies)
    print (total_normal)
    print ('TP: '+ str(TP))
    print ('FP: '+ str(FP))
    print ('FN: '+ str(FN))
    print ('TN: '+ str(TN))
    y_test_int = []
#print confusion_matrix(y_test, answear) #watch out the element in confusion matrix
#convert the result to binary
    for item in range(len(ytest)):
        if ytest[item]==0:
            ytest[item]=1
        if ytest[item]==1:
            ytest[item]=0
        if y_predict[item]==-1:
            y_predict[item]=0
    ytest = [int(numeric_string) for numeric_string in ytest]
   
    precision, recall, thresholds = precision_recall_curve(ytest, y_predict)
    print ("precision: " + str(precision))
    print ("recall: " + str(recall))

    print(clf.decision_function(X))
    #predict_proba = clf.predict_proba(x_test)#the probability of each smple labelled to positive or negative
    #preds = predict_proba[:,1]
    #fpr, tpr, threshold = roc_curve(y_test, preds)
    #print('fpr, tpr, threshold')
    #print(fpr, tpr, threshold)
    #roc_auc = auc(fpr, tpr)
    #plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    #plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')
    #plt.show()

    #X = pca.transform(np.array(data))
# Initialize the two classifiers 
#clf1 = tree.DecisionTreeClassifier(max_depth=5)
#clf2 = RandomForestClassifier(n_estimators=25)
#Initialize SMOTE for the two classifiers
#sm1 = SMOTE(ratio=0.1)
#sm2 = SMOTE(ratio=0.35)

# Send paramters to train the two classifiers
#(y_real_dt, y_pred_dt, y_proba_dt) = TrainClassifier(clf1, sm1, usx, usy, 0.7)
#(y_real_rf, y_pred_rf, y_proba_rf) = TrainClassifier(clf2, sm2, usx, usy, 0.6)

#Decision Tree -- Calculate precision, recall and AUC. Plot the results
#precision_dt, recall_dt, _ = precision_recall_curve(y_real_dt, y_proba_dt)
#fpr_dt, tpr_dt, _ = roc_curve(y_real_dt, y_proba_dt)

#roc_auc_dt = auc(fpr_dt, tpr_dt)
#axes[0].plot(fpr_dt, tpr_dt, label=('DT AUC %.2f' % (roc_auc_dt)))

#f_1_dt = f1_score(y_real_dt, y_pred_dt)
#print "f-score: " + str(f_1_dt) '''
'''labl = "DT F1 %.2f" % (f_1_dt)
axes[1].plot(recall_dt, precision_dt, label=labl, lw=2)

#Random Forest -- Calculate precision, recall and AUC. Plot the results
precision_rf, recall_rf, _ = precision_recall_curve(y_real_rf, y_proba_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_real_rf, y_proba_rf)

roc_auc_rf = auc(fpr_rf, tpr_rf)
axes[0].plot(fpr_rf, tpr_rf, label=('RF AUC %.2f' % (roc_auc_rf)))

f_1_rf = f1_score(y_real_rf, y_pred_rf)
print "f-score: " + str(f_1_rf)
labl = "RF F1 %.2f" % (f_1_rf)
axes[1].plot(recall_rf, precision_rf, label=labl, lw=2)

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc='upper right', fontsize='small')
axes[0].legend(loc='upper right', fontsize='small')

f.tight_layout()
f.savefig('comparison.png')'''



