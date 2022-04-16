"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of ChaosNet on the Statlog (Heart) dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/statlog+(heart)

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Codes import k_cross_validation

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    1 (Absence)           -     0
    2 (Presence)          -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    heart           -   Complete Statlog (Heart) dataset.    
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
    INITIAL_NEURAL_ACTIVITY         -   Initial Neural Activity.
    EPSILON                         -   Noise Intensity.
    DISCRIMINATION_THRESHOLD        -   Discrimination Threshold.
    
    Source: Harikrishnan N.B., Nithin Nagaraj,
    When Noise meets Chaos: Stochastic Resonance in Neurochaos Learning,
    Neural Networks, Volume 143, 2021, Pages 425-435, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.06.025.
    (https://www.sciencedirect.com/science/article/pii/S0893608021002574)
_______________________________________________________________________________


''' 


#import the STATLOG Dataset 
heart = np.array(pd.read_csv('heart.txt', sep=" ", header=None))


#reading data and labels from the dataset
X, y = heart[:,range(0,heart.shape[1]-1)], heart[:,heart.shape[1]-1].astype(str)
y = np.char.replace(y, '1', '0', count=None)
y = np.char.replace(y, '2', '1', count=None)
y = y.astype(float)
y = y.reshape(len(y),1)


#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)



#Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0))/(np.max(X_train,0) - np.min(X_train,0))
X_test_norm = (X_test - np.min(X_test,0))/(np.max(X_test,0) - np.min(X_test,0))



#validation
FOLD_NO = 5
INITIAL_NEURAL_ACTIVITY = [0.08]
DISCRIMINATION_THRESHOLD = [0.06]
EPSILON = [0.17]
k_cross_validation(FOLD_NO, X_train_norm, y_train, X_test_norm, y_test, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)
