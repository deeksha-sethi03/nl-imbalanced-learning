"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to test the efficacy of stand-alone AdaBoost on the Ionosphere dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/ionosphere

"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    b (Bad)      -     0
    g (Good)     -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    ionosphere      -   Complete Ionosphere dataset.    
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
    NEST    -   The maximum number of estimators at which boosting is terminated.  
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
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



#import the IONOSPHERE Dataset 
ionosphere = np.array(pd.read_csv('ionosphere_data.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = ionosphere[:,range(0,ionosphere.shape[1]-1)], ionosphere[:,ionosphere.shape[1]-1].astype(str)


#Norm: B -> 0;  G -> 1
y = y.reshape(len(y),1)
y = np.char.replace(y, 'b', '0', count=None)
y = np.char.replace(y, 'g', '1', count=None)
y = y.astype(int)


#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)



#Normalisation - Column-wise
X_train_norm = X_train
X_train_norm[:,range(2,X_train.shape[1])] = (X_train[:,range(2,X_train.shape[1])]-np.min(X_train[:,range(2,X_train.shape[1])],0))/(np.max(X_train[:,range(2,X_train.shape[1])],0)-np.min(X_train[:,range(2,X_train.shape[1])],0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = X_test
X_test_norm[:,range(2,X_test.shape[1])] = (X_test[:,range(2,X_test.shape[1])]-np.min(X_test[:,range(2,X_test.shape[1])],0))/(np.max(X_test[:,range(2,X_test.shape[1])],0)-np.min(X_test[:,range(2,X_test.shape[1])],0))
X_test_norm = X_test_norm.astype(float)



#Algorithm - AdaBoost
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/' 
    

NEST = np.load(RESULT_PATH+"/h_NEST.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]


clf = AdaBoostClassifier(n_estimators = NEST, random_state=42)
clf.fit(X_train_norm, y_train.ravel())


y_pred = clf.predict(X_test_norm)
f1 = f1_score(y_test, y_pred, average='macro')


print('TRAINING F1 Score', F1SCORE)
print('TESTING F1 Score', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )

