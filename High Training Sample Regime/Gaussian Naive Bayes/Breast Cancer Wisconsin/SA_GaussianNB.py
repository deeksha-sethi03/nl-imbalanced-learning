"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to test the efficacy of stand-alone Gaussian Naive Bayes on the Breast Cancer Wisconsin dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    M (Malignant)      -     0
    B (Benign)         -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    breastcancer    -   Complete Breast Cancer Wisconsin dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).
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


#import the BREAST CANCER WISCONSIN Dataset 
breastcancer = np.array(pd.read_csv('breast-cancer-wisconsin.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = breastcancer[:,range(2,breastcancer.shape[1])], breastcancer[:,1].astype(str)
y = np.char.replace(y, 'M', '0', count=None)
y = np.char.replace(y, 'B', '1', count=None)
y = y.astype(int)
y = y.reshape(len(y),1)
X = X.astype(float)


#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


#Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0))/(np.max(X_train,0) - np.min(X_train,0))
X_test_norm = (X_test - np.min(X_test,0))/(np.max(X_test,0) - np.min(X_test,0))


#Algorithm - Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train_norm, y_train.ravel())


y_pred = clf.predict(X_test_norm)
f1 = f1_score(y_test, y_pred, average='macro')
print('Testing F1 Score = ', f1)

PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/' 
np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
    

