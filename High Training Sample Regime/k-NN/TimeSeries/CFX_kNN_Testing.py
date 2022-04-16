"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to test the efficacy of CFX+k-Nearest Neighbors on the Free Spoken Digit dataset.
Dataset Source: https://github.com/Jakobovski/free-spoken-digit-dataset

"""

import os
import numpy as np
from numpy.fft import fft
from scipy.io import wavfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import ChaosFEX.feature_extractor as CFX

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

    K   -   Number of neighbors to use by default for kneighbors queries.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
_______________________________________________________________________________

Performance metric used:
_______________________________________________________________________________

    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and finds their 
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
    



#Testing
PATH = os.getcwd()
RESULT_PATH = PATH + '/CFX-TUNING/RESULTS/' 
    
INA = np.load(RESULT_PATH+"/h_Q.npy")[0]
EPSILON_1 = np.load(RESULT_PATH+"/h_EPS.npy")[0]
DT = np.load(RESULT_PATH+"/h_B.npy")[0]
K = np.load(RESULT_PATH+"/h_K.npy")[0]

FEATURE_MATRIX_TRAIN = CFX.transform(X_train_norm, INA, 10000, EPSILON_1, DT)
FEATURE_MATRIX_VAL = CFX.transform(X_test_norm, INA, 10000, EPSILON_1, DT)            

clf = KNeighborsClassifier(n_neighbors = K)
clf.fit(FEATURE_MATRIX_TRAIN, y_train.ravel())

y_pred = clf.predict(FEATURE_MATRIX_VAL)
f1 = f1_score(y_test, y_pred, average='macro')
print('TESTING F1 SCORE', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
