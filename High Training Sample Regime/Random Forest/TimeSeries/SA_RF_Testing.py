"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to test the efficacy of stand-alone Random Forest on the Free Spoken Digit dataset.
Dataset Source: https://github.com/Jakobovski/free-spoken-digit-dataset

"""

import os
import numpy as np
from numpy.fft import fft
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    0     -     0
    1     -     1
    2     -     2
    3     -     3
    4     -     4
    5     -     5
    6     -     6
    7     -     7
    8     -     8
    9     -     9
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________

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

    MD      -   The maximum depth of the tree.
    NEST    -   The number of trees in the forest.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
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


#import the TIME SERIES FSDD Dataset 
source = "C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Decision Tree\\TimeSeries\\recordings\\jackson"

#reading data and labels from the dataset
data_length = []
for fileno, filename in enumerate(os.listdir(source)):
    if(fileno<500):       
        sampling_frequency, data = wavfile.read(os.path.join(source,filename))
        if(len(data)>=3000):
            data_length.append(len(data))
            print(filename)
    
y = np.zeros((len(data_length), 1), dtype='int')
input_features = np.min(data_length)
X = np.zeros((len(data_length), input_features))
index = 0
for fileno, filename in enumerate(os.listdir(source)):
    if(fileno<500):       
        sampling_frequency, data = wavfile.read(os.path.join(source,filename))
        print(filename)
        if(len(data)>=3000):
            data_length.append(len(data))
            X[index, :] = np.abs(fft(data[0:input_features]))
            y[index, 0] = filename[0]
            index+=1

#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Normalisation - Row-wise
X_train_norm = (((X_train.T)-np.min(X_train,1))/(np.max(X_train,1)-np.min(X_train,1))).T
X_test_norm = (((X_test.T)-np.min(X_test,1))/(np.max(X_test,1)-np.min(X_test,1))).T
    

#Algorithm - Random Forest
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/' 
    

NEST = np.load(RESULT_PATH+"/h_NEST.npy")[0]
MD = np.load(RESULT_PATH+"/h_MD.npy")[0]


clf = RandomForestClassifier( n_estimators = NEST, max_depth = MD, random_state=42)
clf.fit(X_train_norm, y_train.ravel())


y_pred = clf.predict(X_test_norm)
f1 = f1_score(y_test, y_pred, average='macro')
print('F1 Score', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )

