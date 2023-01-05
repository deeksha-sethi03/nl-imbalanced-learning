#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:39:44 2023

@author: harikrishnan
"""

import os
import numpy as np




# import scipy
from scipy import io

from sklearn import datasets


from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
# from sklearn import datasets
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
from Codes import chaosnet, k_cross_validation
import ChaosFEX.feature_extractor as CFX
from sklearn.model_selection import train_test_split
# from ETC import CCC
import keras
from keras.models import Sequential
from keras.layers import Dense,  Dropout
import pandas as pd

from keras import callbacks

exp_num = 1


# seed_value = 0
# os.environ['PYTHONHASHSEED'] = str(seed_value)
# import random
# random.seed(seed_value)
# np.random.seed(seed_value)




def dnn_2_layer(input_dim, out_dim, batch_size_val, epochs_val, RESULT_PATH, normalized_traindata , y_train):
    
    model = Sequential()

    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(out_dim, activation='softmax'))
    
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpointer = callbacks.ModelCheckpoint(filepath=RESULT_PATH + "checkpoint.hdf5", verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)
    model.fit(normalized_traindata , y_train,
    batch_size=batch_size_val,
    epochs=epochs_val,
    verbose=1,
    callbacks=[checkpointer],
    validation_split=0.3,)
    return model
  


PATH = os.getcwd()


DATA_NAME = 'BNA'

F1_score_Result_array = []






# DL architecture details
input_dim = 16
out_dim = 2
batch_size_val = 32
epochs_val = 100



RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/'


#import the BANK NOTE AUTHENTICATION Dataset 
bank = np.array(pd.read_csv('data_banknote_authentication.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = bank[:,range(0,bank.shape[1]-1)], bank[:,bank.shape[1]-1]
y = y.reshape(len(y),1)
# X = X.astype(float)


#Splitting the dataset for training and testing (80-20)
traindata, testdata, trainlabel, testlabel = train_test_split(X,y,test_size=0.2, random_state=42)

#Normalisation of data [0,1]
traindata=(traindata-np.min(traindata,0))/(np.max(traindata,0)-np.min(traindata,0))
testdata=(testdata-np.min(testdata,0))/(np.max(testdata,0)-np.min(testdata,0))



import ChaosFEX.feature_extractor as CFX

# out : numpy array for dataset with 2 attributes, 2D, float64
#     Contains computed estimates stored as follows:
#         [i,[0,1]] : Firing Rate
#         [i,[2,3]] : Energy
#         [i,[4,5]] : Firing Time
#         [i,[6,7]] : Entropy


INA = 0.080
EPSILON_1 = 0.233
DT = 0.250


traindata = CFX.transform(traindata, INA, 10000, EPSILON_1, DT)
testdata = CFX.transform(testdata, INA, 10000, EPSILON_1, DT)




NUM_CLASS = np.unique(trainlabel).shape[0]
PATH = os.getcwd()
RESULT_PATH_DL = PATH + '/logs/'  + DATA_NAME +'/check-points/'+ str(exp_num) +'/'


try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_DL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_DL)

train_label_ = keras.utils.to_categorical(trainlabel, NUM_CLASS)
   
model = dnn_2_layer(input_dim, out_dim, batch_size_val, epochs_val, RESULT_PATH_DL, traindata, train_label_)
model.summary()
model.load_weights(RESULT_PATH_DL + "checkpoint.hdf5")
y_pred_testdata = np.argmax(model.predict(testdata), axis=-1)
ACC = accuracy_score(testlabel, y_pred_testdata)*100
RECALL = recall_score(testlabel, y_pred_testdata , average="macro")
PRECISION = precision_score(testlabel, y_pred_testdata , average="macro")
F1SCORE = f1_score(testlabel, y_pred_testdata, average="macro")
F1_score_Result_array.append(F1SCORE)
print('TEST: ACCURACY = ',ACC, " F1 SCORE = ",F1SCORE)
    
    
# Final Result Plot

RESULT_PATH_FINAL = PATH + '/' +'CFX+DL-RESULTS' + '/'+ DATA_NAME + '/' + str(exp_num) +'/'


F1_score_Result_array_npy = np.array(F1_score_Result_array)

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)



# Saving Results



np.save(RESULT_PATH_FINAL +'F1-score.npy', F1_score_Result_array_npy)


