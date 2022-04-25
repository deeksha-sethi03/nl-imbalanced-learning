"""
Created on Tue Feb 22 12:17:06 2022

Author: Harikrishnan NB (harikrishnannb07@gmail.com)
Code Description: A python code to  demonstrate the overall performance change
of the Wine dataset in the low training sample regime after employing the CFX 
features.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

'''
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    F1SCORE_FINAL_MATRIX_SA     -   F1 score matrix obtained from the low 
                                    training sample regime code of stand-alone 
                                    algorithms for the dataset.
    F1SCORE_FINAL_MATRIX_CFX    -   F1 score matrix obtained from the low 
                                    training sample regime code of CFX+ML 
                                    algorithms for the dataset.
    Change                      -   Performance change percentage in CFX+ML 
                                    algorithms with respect to the stand-alone 
                                    algorithms.
    neg_count                   -   Number of samples per class for which the 
                                    stand-alone algorithm is performing better 
                                    than the CFX+ML algorithm.
    pos_count                   -   Number of samples per class for which the 
                                    CFX+ML algorithm is performing better 
                                    than the stand-alone algorithm.
    min_max_array               -   A array of all the minimum and maximum 
                                    performance boost change percentage only for
                                    cases with an increase in performance after
                                    employing the CFX features.
_______________________________________________________________________________

'''

labels = ['Decision Tree','Random Forest','AdaBoost','SVM', 'k-NN', 'ChaosNet', 'Gaussian NB']
PATH = os.getcwd()
RESULT_PATH_SA = PATH + '/lOW TRAINING RESULTS/' 
F1SCORE_FINAL_MATRIX_SA = np.load(RESULT_PATH_SA+"/SA_F1_SCORES.npy")



labels = ['CFX + Decision Tree','CFX + Random Forest','CFX + AdaBoost','CFX + SVM', 'CFX + k-NN', 'ChaosNet', 'CFX + Gaussian NB']
PATH = os.getcwd()
RESULT_PATH_CFX = PATH + '/LOW TRAINING RESULTS/' 
F1SCORE_FINAL_MATRIX_CFX = np.load(RESULT_PATH_CFX+"/CFX_F1_SCORES.npy")

# Performance  Change

Change = 100*(F1SCORE_FINAL_MATRIX_CFX - F1SCORE_FINAL_MATRIX_SA)/F1SCORE_FINAL_MATRIX_SA
min_max_array = np.zeros((2, Change.shape[1]))

for col in range(0, Change.shape[1]):
    list1 = Change[:, col]
      
    neg_count = len(list(filter(lambda x: (x < 0), list1)))
      
    # we can also do len(list1) - neg_count
    pos_count = len(list(filter(lambda x: (x >= 0), list1)))
      
    print("Positive numbers in the list: ", pos_count)
    print("Negative numbers in the list: ", neg_count)
    
    
    temp_change = []
    if pos_count >= neg_count:
        for row in range(0, Change.shape[0]):
            if Change[row,col] >= 0:
                temp_change.append(round(Change[row, col], 2))
                
        min_max_array[0, col] = np.min(temp_change)
        min_max_array[1, col] = np.max(temp_change)
        print(labels[col],"=",min_max_array[:, col])
        print("******************************")
print(min_max_array)
