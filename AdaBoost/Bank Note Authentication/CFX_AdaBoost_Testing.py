"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to test the efficacy of CFX+AdaBoost on the Bank Note Authentication dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
import ChaosFEX.feature_extractor as CFX

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

CFX hyperparameter description:
_______________________________________________________________________________
    INA         -   Initial Neural Activity
    EPSILON_1   -   Noise Intensity
    DT          -   Discrimination Threshold
        
    Source: Harikrishnan N.B., Nithin Nagaraj,
    When Noise meets Chaos: Stochastic Resonance in Neurochaos Learning,
    Neural Networks, Volume 143, 2021, Pages 425-435, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.06.025.
    (https://www.sciencedirect.com/science/article/pii/S0893608021002574)

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


#Testing
PATH = os.getcwd()
RESULT_PATH = PATH + '/CFX-TUNING/RESULTS/' 



INA = np.load(RESULT_PATH+"/h_Q.npy")[0]
EPSILON_1 = np.load(RESULT_PATH+"/h_EPS.npy")[0]
DT = np.load(RESULT_PATH+"/h_B.npy")[0]
NEST = np.load(RESULT_PATH+"/h_NEST.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]


FEATURE_MATRIX_TRAIN = CFX.transform(X_train_norm, INA, 10000, EPSILON_1, DT)
FEATURE_MATRIX_VAL = CFX.transform(X_test_norm, INA, 10000, EPSILON_1, DT)            
clf = AdaBoostClassifier(n_estimators=NEST, random_state=42)
clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

y_pred = clf.predict(FEATURE_MATRIX_VAL)
f1 = f1_score(y_test, y_pred, average='macro')


print('TRAINING F1 SCORE', F1SCORE)
print('TESTING F1 SCORE', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
