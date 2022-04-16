"""
Created on Mon Dec 13 04:54:36 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Dataset: Haberman's Survival
Code Description:  
In low training sample regime, 150 random trials for training with [1,9] data 
instances per class are considered. Seven algorithms namely, Decision Tree, 
Random Forest, AdaBoost, Support Vector Machine (SVM), k-Nearest Neighbors (KNN), 
ChaosNet and Gaussian Naive Bayes (GNB) are tested in the low training sample 
regime on this dataset. The trends of all algorithms in the high training sample 
regime are plotted through this code. The performance metric used throughout all 
experiments in this code is Macro F1-score. All values in this code are obtained 
from the results saved by the corresponding SA_LTS.py file for this dataset
    
"""
import os
import numpy as np
import matplotlib.pyplot as plt


markers = ['->k', '--g', '-.b','-om', '-sy','-^c','-*r']
labels = ['Decision Tree','Random Forest','AdaBoost','SVM', 'k-NN', 'ChaosNet', 'Gaussian NB']
PATH = os.getcwd()
RESULT_PATH = PATH + '/LOW TRAINING RESULTS/' 
F1SCORE_FINAL_MATRIX = np.load(RESULT_PATH+"/SA_F1_SCORES.npy")



plt.figure(figsize=(15,10))
low_samples_per_class = 9
trials = np.arange(1, low_samples_per_class+1)
for x in range(0, 7):
    plt.plot(trials, F1SCORE_FINAL_MATRIX[:, x], markers[x], markersize = 10, label = labels[x])

#my_xticks = ['T1', 'T2', 'T3', 'T4', 'T5']
#plt.xticks(task_array, my_xticks, fontsize=25)
plt.xticks(trials, fontsize=50)
plt.yticks(fontsize=50)
plt.grid(True)
plt.xlabel('Number of samples per class', fontsize=50)
plt.ylabel('Average F1-score', fontsize=50)
plt.ylim(-0.6, 0.8)
plt.tight_layout()
plt.legend(fontsize = 35)

plt.savefig(RESULT_PATH+"/SA-HABERMAN.eps", format='eps', dpi=300)
plt.savefig(RESULT_PATH+"SA-HABERMAN.jpg", format='jpg', dpi=300)