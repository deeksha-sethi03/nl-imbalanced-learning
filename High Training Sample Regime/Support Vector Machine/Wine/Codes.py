# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Dtd: 22 Dec. 2020
ChaosNet decision function
"""

import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import ChaosFEX.feature_extractor as CFX
from sklearn.svm import SVC




def k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON):
    """

    Parameters
    ----------
    FOLD_NO : TYPE-Integer
        DESCRIPTION-K fold classification.
    traindata : TYPE-numpy 2D array
        DESCRIPTION - Traindata
    trainlabel : TYPE-numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE-numpy 2D array
        DESCRIPTION - Testdata
    testlabel : TYPE - numpy 2D array
        DESCRIPTION - Testlabel
    INITIAL_NEURAL_ACTIVITY : TYPE - numpy 1D array
        DESCRIPTION - initial value of the chaotic skew tent map.
    DISCRIMINATION_THRESHOLD : numpy 1D array
        DESCRIPTION - thresholds of the chaotic map
    EPSILON : TYPE numpy 1D array
        DESCRIPTION - noise intenity for NL to work (low value of epsilon implies low noise )

    Returns
    -------
    FSCORE, Q, B, EPS, EPSILON

    """
    
    BESTF1 = 0
    KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
    KF.get_n_splits(traindata) 
    print(KF) 
    
    
    for DT in DISCRIMINATION_THRESHOLD:
        
        for INA in INITIAL_NEURAL_ACTIVITY:
            
            for EPSILON_1 in EPSILON:
                
                for c in np.arange(0.1, 100.0, 0.1):                
                    FSCORE_TEMP=[]
            
                    for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                        
                        X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                        Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
            
    
                        # Extract features
                        FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                        FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)            
                        clf = SVC(C = c, kernel='rbf', decision_function_shape='ovr', random_state=42)
                        clf.fit(FEATURE_MATRIX_TRAIN, Y_TRAIN.ravel())
                        Y_PRED = clf.predict(FEATURE_MATRIX_VAL)
                        F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                        FSCORE_TEMP.append(F1SCORE)
                        print(F1SCORE)
                    print("Mean F1-Score for Q = ", INA,"B = ", DT,"EPSILON = ", EPSILON_1,"C = ", c," is  = ",  np.mean(FSCORE_TEMP)  )
    
                    if(np.mean(FSCORE_TEMP) > BESTF1):
                        BESTF1 = np.mean(FSCORE_TEMP)
                        BESTINA = INA
                        BESTDT = DT
                        BESTEPS = EPSILON_1
                        BESTC = c
                            
    
    
    
    print("Saving Hyperparameter Tuning Results")
    
    
    PATH = os.getcwd()
    RESULT_PATH = PATH + '/CFX-TUNING/RESULTS/'
    
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s not required" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    np.save(RESULT_PATH+"/h_Q.npy", np.array([BESTINA]) ) 
    np.save(RESULT_PATH+"/h_B.npy", np.array([BESTDT]) )
    np.save(RESULT_PATH+"/h_EPS.npy", np.array([BESTEPS]) )
    np.save(RESULT_PATH+"/h_C.npy", np.array([BESTC]) )
    np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) )

    
    
    
    print("BEST F1SCORE", BESTF1)
    print("BEST INITIAL NEURAL ACTIVITY = ", BESTINA)
    print("BEST DISCRIMINATION THRESHOLD = ", BESTDT)
    print("BEST EPSILON = ", BESTEPS)
    print("BEST C = ", BESTC)
    

    




