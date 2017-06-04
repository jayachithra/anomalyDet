
# coding: utf-8

# Anomaly Detection Lab!
# Group 8: Azqa Nadeem, Jaya Chithra

get_ipython().magic(u'matplotlib inline')

import datetime
import time
import pandas as pd
from pandas import DataFrame, read_csv
from sklearn import svm
import statsmodels.api as api
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from operator import itemgetter
from itertools import groupby
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition
from matplotlib import cm as cm

def fit_model(series, order):
    tr_min = 374401
    tr_max = 460800
    te_min, te_max =50401,136800 

    newdata = df[series][tr_min:tr_max]
    newtime = df.Timestamp[tr_min:tr_max]
    DataFrame(newdata).plot()
    testdata = test[series][te_min:te_max]
    testtime = test.Timestamp[te_min:te_max]

    dat = DataFrame(data={'timestamp' : pd.to_datetime(newtime, dayfirst=True).values, series : newdata})
    dat.set_index("timestamp", inplace=True)
    dat_test = DataFrame(data={'timestamp' : pd.to_datetime(testtime, dayfirst=True).values, series : testdata})
    dat_test.set_index("timestamp", inplace=True)

   
    model = api.tsa.ARMA(dat, order)
    model_fit = model.fit(trend='nc', disp=False)
    model = api.tsa.ARMA(dat_test, order)
    model_test = model.fit(trend='nc', disp=False)

    print model_fit.summary()
    return (model_test)

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

def correlation_matrix(df, labels, filename):


    fig = plt.figure(figsize=(20, 10))
    '''ax1 = fig.add_subplot(111)
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
    plt.show()'''
    df.plot(kind='box', title=filename)
    #plt.savefig('normed_'+filename)

def normalize(x, blah):
       mean = x.mean()
       centered= []
       for i in x:
           centered.append(np.float64(x)-np.float64(mean))
       
       maxi = centered.max()
       mini = centered.min()
       normalizedData = []
       for i in centered:
           normalizedData.append((float(i)-mini)/(maxi-mini))
       return normalizedData

def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%d/%m/%Y %H:%M:%S %p')
    return time.mktime(time_stamp)

def ParseData(filename):
    ah = open(filename, 'r')
    data = []#contains features
    y = []#contains labels
    
    ah.readline()#skip first line
    for line_ah in ah:

        timestamp = string_to_timestamp(line_ah.strip().split(',')[0])# date
        fit101 = float(line_ah.strip().split(',')[1])
        lit101 = float(line_ah.strip().split(',')[2])
        mv101 = float(line_ah.strip().split(',')[3])
        p101 = float(line_ah.strip().split(',')[4] )
        
        ait201 = float(line_ah.strip().split(',')[6])
        ait202 = float(line_ah.strip().split(',')[7])
        ait203 = float(line_ah.strip().split(',')[8])
        fit201 = float(line_ah.strip().split(',')[9])
        mv201 = float(line_ah.strip().split(',')[10])
        p203 = float(line_ah.strip().split(',')[13])
        p205 = float(line_ah.strip().split(',')[15])
   
        dpit301 = float(line_ah.strip().split(',')[17])
        fit301 = float(line_ah.strip().split(',')[18])
        lit301 = float(line_ah.strip().split(',')[19])
        mv301 = float(line_ah.strip().split(',')[20])
        mv302 = float(line_ah.strip().split(',')[21])
        mv303 = float(line_ah.strip().split(',')[22])
        mv304 = float(line_ah.strip().split(',')[23])
        p301 = float(line_ah.strip().split(',')[24])
        p302 = float(line_ah.strip().split(',')[25])
        
        ait401 = float(line_ah.strip().split(',')[26])
        ait402 = float(line_ah.strip().split(',')[27])
        fit401 = float(line_ah.strip().split(',')[28])
        lit401 = float(line_ah.strip().split(',')[29])

        ait501 = float(line_ah.strip().split(',')[35])
        ait502 = float(line_ah.strip().split(',')[36])
        ait503 = float(line_ah.strip().split(',')[37])
        ait504 = float(line_ah.strip().split(',')[38])
        fit501 = float(line_ah.strip().split(',')[39])
        fit502 = float(line_ah.strip().split(',')[40])
        fit503 = float(line_ah.strip().split(',')[41])
        fit504 = float(line_ah.strip().split(',')[42])
        pit501 = float(line_ah.strip().split(',')[45])
        pit502 = float(line_ah.strip().split(',')[46])
        pit503 = float(line_ah.strip().split(',')[47])

        fit601 = float(line_ah.strip().split(',')[48])
        p602 = float(line_ah.strip().split(',')[50])
        

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
                     fit601, p602]) 
    return data


if __name__ == "__main__":
    src = 'normalize_normal.csv'
    testFile = 'normalize_attack.csv'

    ####################### Correlation ################################
    test = read_csv(testFile)    
    df = read_csv(src,squeeze=True)
    '''
    df.FIT101 = normalize(df.FIT101, test.FIT101)
    df.LIT101 = normalize(df.LIT101, test.LIT101)
    df.MV101 = normalize(df.MV101, test.MV101)
    df.P101 =  normalize(df.P101, test.P101)

    df.AIT201 = normalize(df.AIT201, test.AIT201)
    df.AIT202 = normalize(df.AIT202, test.AIT202)
    df.AIT203 = normalize(df.AIT203, test.AIT203)
    df.FIT201 =  normalize(df.FIT201, test.FIT201)
    df.MV201 =  normalize(df.MV201, test.MV201)
    df.P203 =  normalize(df.P203, test.P203)
    df.P205 =  normalize(df.P205, test.P205)

    df.DPIT301 = normalize(df.DPIT301, test.DPIT301)
    df.FIT301 = normalize(df.FIT301, test.FIT301)
    df.LIT301 = normalize(df.LIT301, test.LIT301)
    df.MV301 =  normalize(df.MV301, test.MV301)
    df.MV302 =  normalize(df.MV302, test.MV302)
    df.MV303 =  normalize(df.MV303, test.MV303)
    df.MV304 =  normalize(df.MV304, test.MV304)
    df.P301 =  normalize(df.P301, test.P301)
    df.P302 =  normalize(df.P302, test.P302)

    df.AIT401 =  normalize(df.AIT401, test.AIT401)
    df.AIT402 =  normalize(df.AIT402, test.AIT402)
    df.FIT401 =  normalize(df.FIT401, test.FIT401)
    df.LIT401 =  normalize(df.LIT401, test.LIT401)

    df.AIT501 = normalize(df.AIT501, test.AIT501)
    df.AIT502 = normalize(df.AIT502, test.AIT502)
    df.AIT503 = normalize(df.AIT503, test.AIT503)
    df.AIT504 =  normalize(df.AIT504, test.AIT504)
    df.FIT501 =  normalize(df.FIT501, test.FIT501)
    df.FIT502 =  normalize(df.FIT502, test.FIT502)
    df.FIT503 =  normalize(df.FIT503, test.FIT503)
    df.FIT504 =  normalize(df.FIT504, test.FIT504)
    df.PIT501 =  normalize(df.PIT501, test.PIT501)
    df.PIT502 =  normalize(df.PIT502, test.PIT502)
    df.PIT503 =  normalize(df.PIT503, test.PIT503)

    df.FIT601 =  normalize(df.FIT601, test.FIT601)
    df.P602 =  normalize(df.P602, test.P602)'''

    #test.to_csv('normalize_attack.csv', sep=',')

    
    '''p1 = {'FIT101': df.FIT101, 'LIT101': df.LIT101, 'MV101': df.MV101, 'P101': df.P101}
    p1_labels = ['FIT101', 'LIT101', 'MV101', 'P101']
    p2 = {'AIT201': df.AIT201, 'AIT202': df.AIT202, 'AIT203': df.AIT203, 'FIT201' :df.FIT201, 'MV201': df.MV201, 'P203' :df.P203, 'P205' : df.P205}
    p2_labels = ['AIT201', 'AIT202', 'AIT203', 'FIT201','MV201', 'P203', 'P205' ]
    p3 = {'DPIT301' : df.DPIT301, 'FIT301': df.FIT301, 'LIT301' : df.LIT301, 'MV301' : df.MV301, 'MV302' : df.MV302, 'MV303' : df.MV303, 'MV304' : df.MV304, 'P301' :df.P301, 'P302' : df.P302}
    p3_labels= ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302']
    p4 = {'AIT401' : df.AIT401, 'AIT402': df.AIT402, 'FIT401' : df.FIT401, 'LIT401': df.LIT401}
    p4_labels = ['AIT401', 'AIT402', 'FIT401','LIT401' ]
    p5 = {'AIT501':df.AIT501, 'AIT502': df.AIT502, 'AIT503': df.AIT503, 'AIT504': df.AIT504, 'FIT501': df.FIT501, 'FIT502': df.FIT502, 'FIT503': df.FIT503, 'FIT504' : df.FIT504, 'PIT501': df.PIT501, 'PIT502': df.PIT502, 'PIT503': df.PIT503}
    p5_labels = ['AIT501', 'AIT502', 'AIT503' , 'AIT504', 'FIT501', 'FIT502', 'FIT503' ,'FIT504', 'PIT501', 'PIT502', 'PIT503']
    p6 = {'FIT601':df.FIT601, 'P602': df.P602}
    p6_labels = ['FIT601', 'P602']
    
    
    correlation_matrix(DataFrame(data=p1), p1_labels, 'p1.png')
    correlation_matrix(DataFrame(data=p2), p2_labels, 'p2.png')
    correlation_matrix(DataFrame(data=p3), p3_labels, 'p3.png')
    correlation_matrix(DataFrame(data=p4), p4_labels, 'p4.png')
    correlation_matrix(DataFrame(data=p5), p5_labels, 'p5.png')
    correlation_matrix(DataFrame(data=p6), p6_labels, 'p6.png')'''

    '''features = p1
    features.update(p2)
    features.update(p3)
    features.update(p4)
    features.update(p5)
    features.update(p6)
    
    p1_labels.extend(p2_labels)
    p1_labels.extend(p3_labels)
    p1_labels.extend(p4_labels)
    p1_labels.extend(p5_labels)
    p1_labels.extend(p6_labels)
    correlation_matrix(DataFrame(data=p1), p1_labels, 'all_normalized.png')'''
    #############################################################

    trainingData = ParseData(src)
    testData = ParseData(testFile)
    
    print 'done reading'
    ######################   PCA  ######################################
    '''pca = decomposition.PCA() #  for dim red
    x_train = pca.fit(np.array(trainingData))
    x_test = x_train.transform(testData)
    normal, abnormal =x_test[:,0]**2, x_test[:,5]**2
    for i in xrange(1,4):
        normal += x_test[:,i]**2
    for i in xrange(6,37):
        abnormal += x_test[:,i]**2
      
    DataFrame(normal).plot(kind='line')
    DataFrame(abnormal).plot(kind='line')
    

    predicted = ((normal > 6) | (abnormal>29))

    #f = open('timestamps.txt', 'w')
    TP, TN, FP, FN  =0.0,0.0,0.0,0.0
    normals, attacks = 0.0,0.0
    for i in xrange(0, len(predicted)):
        if ((test['Normal/Attack'][i]=='Normal') and (predicted[i]==False)):
		   TN+=1
        if ((test['Normal/Attack'][i]=='Normal') and (predicted[i]==True)):
		   FP+=1
        if ((test['Normal/Attack'][i]=='Attack') and (predicted[i]==True)):
		   TP+=1
                   #print >> f,'\n', test['Timestamp'][i]
        if ((test['Normal/Attack'][i]=='Attack') and (predicted[i]==False)):
		   FN+=1
        if(test['Normal/Attack'][i]=='Attack'):
          attacks+=1
        if(test['Normal/Attack'][i]=='Normal'):
          normals+=1
    print "True positives: "+ str(TP)
    print "False positives: "+ str(FP)
    print "True Negatives: "+ str(TN)
    print "False Negatives: "+ str(FN)
    print "normals: "+ str(normals)
    print "attacks: "+ str(attacks)
    precision = TP/(TP+FP)
    recall =TP/ (TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    print f1'''


    ################################################################################3

    # random sampling for testing
    #idx = random.sample(xrange(0, len(df.FIT101)),len(df.FIT101)/6)
    #idx.sort()
    tr_min = 374401
    tr_max = 460800
    te_min, te_max =50401,136800 

    '''model_FIT101 = fit_model('FIT101', (4,2))
    model_LIT101 = fit_model('LIT101', (4,2))
    model_MV101 = fit_model('MV101', (1,0))

    model_AIT201 = fit_model('AIT201', (4,1))
    model_AIT202 = fit_model('AIT202', (4,2))
    model_AIT203 = fit_model('AIT203', (4,2))
    model_FIT201 = fit_model('FIT201', (4,2))
    model_MV201 = fit_model('MV201', (4,2))

    model_DPIT301 = fit_model('MV201', (4,2))
    model_FIT301 = fit_model('MV201', (4,2))
    model_LIT301 = fit_model('MV201', (4,2))
    model_MV301 = fit_model('MV201', (4,2))
    model_MV302 = fit_model('MV201', (4,2))
    model_MV303 = fit_model('MV201', (4,2))
    model_MV304 = fit_model('MV201', (4,2))
    model_MV201 = fit_model('MV201', (4,2))'''

    

    '''# get what you need for predicting one-step ahead
    params = model_fit.params
    residuals = model_fit.resid
    p = model_fit.k_ar
    q = model_fit.k_ma
    k_exog = model_fit.k_exog
    k_trend = model_fit.k_trend
    steps = 1

    output = _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=test.FIT101, exog=None, start=len(test.FIT101))'''
    




