"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of CFX+Decision Tree on the Free Spoken Digit dataset.
Dataset Source: https://github.com/Jakobovski/free-spoken-digit-dataset

"""

import os
import numpy as np
from numpy.fft import fft
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from Codes import k_cross_validation

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
    

#validation
FOLD_NO = 5
INITIAL_NEURAL_ACTIVITY = [0.340]
DISCRIMINATION_THRESHOLD = [0.499]
EPSILON = [0.178]
k_cross_validation(FOLD_NO, X_train_norm, y_train, X_test_norm, y_test, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)
