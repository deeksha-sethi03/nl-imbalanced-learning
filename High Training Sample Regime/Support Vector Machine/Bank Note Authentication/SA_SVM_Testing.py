"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to test the efficacy of stand-alone SVM on the Bank Note Authentication dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/' 
    

c = np.load(RESULT_PATH+"/h_C.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]


clf = SVC(C = c, kernel='rbf', decision_function_shape='ovr', random_state = 42)
clf.fit(X_train_norm, y_train.ravel())


y_pred = clf.predict(X_test_norm)
f1 = f1_score(y_test, y_pred, average='macro')


print('Training F1 Score = ', F1SCORE)
print('Testing F1 Score = ', f1)


np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
