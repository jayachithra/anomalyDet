
# coding: utf-8

# Anomaly Detection Lab!
# Group 8: Azqa Nadeem, Jaya Chithra

get_ipython().magic(u'matplotlib inline')

import datetime
import time
import pandas as pd
from pandas import DataFrame, read_csv
from sklearn import svm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
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
import numpy as np
from matplotlib import pyplot as plt
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

def normalize(maxmin, x):
       maxi = maxmin.max()
       mini = maxmin.min()
       normalizedData = []
       for i in x:
           normalizedData.append((float(i)-mini)/(maxi-mini))
       return normalizedData

def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%d/%m/%Y %H:%M:%S %p')
    return time.mktime(time_stamp)

# Training the classifier and K-Fold cross-validation
def TrainClassifier(clf, sm, usx, usy, cutoff):
	total_frauds = 0
	non_frauds = 0
	TP, FP, FN, TN = 0, 0, 0, 0 #Per fold values
	aTP, aFP, aFN, aTN = 0, 0, 0, 0 #Average values
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
	    for i in xrange(len(y_predict)):
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
            # collect y_test, preds and y_predict of all folds
	    y_real.append(y_test)
	    y_proba.append(preds)
	    y_pred.append(y_predict)

        # calculate average FP, TP, FN, TN
	aTP = TP/10
	aFP = FP/10
	aFN = FN/10
	aTN = TN/10
	print 'Average TP: '+ str(aTP)
	print 'Average FP: '+ str(aFP)
	print 'Average FN: '+ str(aFN)
	print 'Average TN: '+ str(aTN)
	print 'Average actual frauds: ' + str(total_frauds/10)
	print 'Average actual legits: ' + str(non_frauds/10)
	    
	y_real = np.concatenate(y_real)
	y_pred = np.concatenate(y_pred)
	y_proba= np.concatenate(y_proba)
        return (y_real, y_pred, y_proba)

if __name__ == "__main__":
    src = 'normalize_normal.csv'
    testFile = 'normalize_attack.csv'

    ####################### Correlation ################################
    test = read_csv(testFile)    
    df = read_csv(src)
    '''
    test.FIT101 = normalize(df.FIT101, test.FIT101)
    test.LIT101 = normalize(df.LIT101, test.LIT101)
    test.MV101 = normalize(df.MV101, test.MV101)
    test.P101 =  normalize(df.P101, test.P101)

    test.AIT201 = normalize(df.AIT201, test.AIT201)
    test.AIT202 = normalize(df.AIT202, test.AIT202)
    test.AIT203 = normalize(df.AIT203, test.AIT203)
    test.FIT201 =  normalize(df.FIT201, test.FIT201)
    test.MV201 =  normalize(df.MV201, test.MV201)
    test.P203 =  normalize(df.P203, test.P203)
    test.P205 =  normalize(df.P205, test.P205)

    test.DPIT301 = normalize(df.DPIT301, test.DPIT301)
    test.FIT301 = normalize(df.FIT301, test.FIT301)
    test.LIT301 = normalize(df.LIT301, test.LIT301)
    test.MV301 =  normalize(df.MV301, test.MV301)
    test.MV302 =  normalize(df.MV302, test.MV302)
    test.MV303 =  normalize(df.MV303, test.MV303)
    test.MV304 =  normalize(df.MV304, test.MV304)
    test.P301 =  normalize(df.P301, test.P301)
    test.P302 =  normalize(df.P302, test.P302)

    test.AIT401 =  normalize(df.AIT401, test.AIT401)
    test.AIT402 =  normalize(df.AIT402, test.AIT402)
    test.FIT401 =  normalize(df.FIT401, test.FIT401)
    test.LIT401 =  normalize(df.LIT401, test.LIT401)

    test.AIT501 = normalize(df.AIT501, test.AIT501)
    test.AIT502 = normalize(df.AIT502, test.AIT502)
    test.AIT503 = normalize(df.AIT503, test.AIT503)
    test.AIT504 =  normalize(df.AIT504, test.AIT504)
    test.FIT501 =  normalize(df.FIT501, test.FIT501)
    test.FIT502 =  normalize(df.FIT502, test.FIT502)
    test.FIT503 =  normalize(df.FIT503, test.FIT503)
    test.FIT504 =  normalize(df.FIT504, test.FIT504)
    test.PIT501 =  normalize(df.PIT501, test.PIT501)
    test.PIT502 =  normalize(df.PIT502, test.PIT502)
    test.PIT503 =  normalize(df.PIT503, test.PIT503)

    test.FIT601 =  normalize(df.FIT601, test.FIT601)
    test.P602 =  normalize(df.P602, test.P602)

    test.to_csv('normalize_attack.csv', sep=',')'''

    '''
    p1 = {'FIT101': df.FIT101, 'LIT101': df.LIT101, 'MV101': df.MV101, 'P101': df.P101}
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
    correlation_matrix(DataFrame(data=p6), p6_labels, 'p6.png')

    features = p1
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

    
    ah = open(src, 'r')
    X = []#contains features
    y = []#contains labels
    data = []
    color = []
    
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

    print 'done reading'
    '''x_mean = data;
    des = 'normal_data.csv'
    ch_dfa = open(des,'w')

    sentence = []
    for i in range(len(x_mean)):
        for j in range(len(x_mean[i])):
            sentence.append(str(x_mean[i][j]))
        sentence.append(str(y[i]))
        ch_dfa.write(' '.join(sentence))
        ch_dfa.write('\n')
        sentence=[]
        ch_dfa.flush()   

    pca = decomposition.PCA() # n_components=3 for dim red
    X = pca.fit_transform(np.array(data))'''
    #x_train, x_test = train_test_split(X, test_size = 0.2)#test_size:
    #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    #clf.fit(x_train)
    #y_pred_train = clf.predict(x_train)
    #y_pred_test = clf.predict(x_test)

    
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
#print "f-score: " + str(f_1_dt)
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



