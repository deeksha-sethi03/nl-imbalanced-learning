"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of stand-alone AdaBoost on the Haberman's Survival dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/haberman's+survival

"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    1 (< 5yrs)      -     0
    2 (>= 5yrs)     -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    haberman        -   Complete Haberman's Survival dataset.    
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


#import the HABERMAN'S SURVIVAL Dataset 
haberman = np.array(pd.read_csv('haberman.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = haberman[:,range(0,haberman.shape[1]-1)], haberman[:,haberman.shape[1]-1].astype(str)
y = np.char.replace(y, '1', '0', count=None)
y = np.char.replace(y, '2', '1', count=None)
y = y.astype(int)
y = y.reshape(len(y),1)


#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


#Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0))/(np.max(X_train,0) - np.min(X_train,0))
X_test_norm = (X_test - np.min(X_test,0))/(np.max(X_test,0) - np.min(X_test,0))



#Algorithm - AdaBoost

n_estimator = [1, 10, 50, 100, 500, 1000, 5000, 10000]
BESTF1 = 0
FOLD_NO = 5
KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
KF.get_n_splits(X_train_norm) 
print(KF) 
for NEST in n_estimator:
    
                
    FSCORE_TEMP=[]

    for TRAIN_INDEX, VAL_INDEX in KF.split(X_train_norm):
        
        X_TRAIN, X_VAL = X_train_norm[TRAIN_INDEX], X_train_norm[VAL_INDEX]
        Y_TRAIN, Y_VAL = y_train[TRAIN_INDEX], y_train[VAL_INDEX]
    
        
        clf = AdaBoostClassifier(n_estimators=NEST, random_state=42)
        clf.fit(X_TRAIN, Y_TRAIN.ravel())
        Y_PRED = clf.predict(X_VAL)
        f1 = f1_score(Y_VAL, Y_PRED, average='macro')
        FSCORE_TEMP.append(f1)
        print('F1 Score', f1)
    print("Mean F1-Score for N-EST = ", NEST," is  = ",  np.mean(FSCORE_TEMP)  )
    if(np.mean(FSCORE_TEMP) > BESTF1):
        BESTF1 = np.mean(FSCORE_TEMP)
        BESTNEST = NEST
        
        
print("BEST F1SCORE", BESTF1)
print("BEST NEST = ", BESTNEST)




print("Saving Hyperparameter Tuning Results")
   
  
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/'


try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

np.save(RESULT_PATH+"/h_NEST.npy", np.array([BESTNEST]) ) 
np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) ) 
