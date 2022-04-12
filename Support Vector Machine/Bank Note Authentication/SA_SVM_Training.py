"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of stand-alone SVM on the Bank Note Authentication dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    0 (Genuine)      -     0
    1 (Forgery)      -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    bank            -   Complete Bank Note Authentication dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).

_______________________________________________________________________________

ML hyperparameter description:
_______________________________________________________________________________

    c   -   Regularization parameter. The strength of the regularization is 
    inversely proportional to C. Must be strictly positive. The penalty is a 
    squared l2 penalty.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
_______________________________________________________________________________

Performance metric used:
_______________________________________________________________________________

    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and find stheir 
    unweighted mean. This does not take label imbalance into account.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
_______________________________________________________________________________

'''


#import the BANK NOTE AUTHENTICATION Dataset 
bank = np.array(pd.read_csv('data_banknote_authentication.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = bank[:,range(0,bank.shape[1]-1)], bank[:,bank.shape[1]-1]
y = y.reshape(len(y),1)
# X = X.astype(float)


#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


#Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0))/(np.max(X_train,0) - np.min(X_train,0))
X_test_norm = (X_test - np.min(X_test,0))/(np.max(X_test,0) - np.min(X_test,0))


#Algorithm - SVM
BESTF1 = 0
FOLD_NO = 5
KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
KF.get_n_splits(X_train_norm) 
print(KF) 
for c in np.arange(0.1, 100.0, 0.1):
    FSCORE_TEMP=[]
    
    for TRAIN_INDEX, VAL_INDEX in KF.split(X_train_norm):
        
        X_TRAIN, X_VAL = X_train_norm[TRAIN_INDEX], X_train_norm[VAL_INDEX]
        Y_TRAIN, Y_VAL = y_train[TRAIN_INDEX], y_train[VAL_INDEX]
    

        clf = SVC(C = c, kernel='rbf', decision_function_shape='ovr', random_state=42)
        clf.fit(X_TRAIN, Y_TRAIN.ravel())
        Y_PRED = clf.predict(X_VAL)
        f1 = f1_score(Y_VAL, Y_PRED, average='macro')
        FSCORE_TEMP.append(f1)
        print('F1 Score', f1)
    print("Mean F1-Score for C = ", c," is  = ",  np.mean(FSCORE_TEMP)  )
    if(np.mean(FSCORE_TEMP) > BESTF1):
        BESTF1 = np.mean(FSCORE_TEMP)
        BESTC = c
        
print("BEST F1SCORE", BESTF1)
print("BEST C = ", BESTC)


print("Saving Hyperparameter Tuning Results")
   
  
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/'


try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

np.save(RESULT_PATH+"/h_C.npy", np.array([BESTC]) ) 
np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) ) 

    