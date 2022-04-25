"""
Created on Tue Feb 22 14:26:48 2022

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to plot the results of all algorithms (SA and CFX+ML)
in the high training sample regime for the Free Spoken Digit dataset.

"""
import os
import numpy as np
import matplotlib.pyplot as plt

 
PATH = os.getcwd()
DATA_NAME = "TimeSeries"

RESULT_PATH = 'C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\BARPLOT_RESULTS/'
    
    
try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)


#DECISION TREE
SA_DT = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Decision Tree\\"+DATA_NAME+"\\SA-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]
CFX_DT = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Decision Tree\\"+DATA_NAME+"\\CFX-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]


#RANDOM FOREST
SA_RF = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Random Forest\\"+DATA_NAME+"\\SA-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]
CFX_RF = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Random Forest\\"+DATA_NAME+"\\CFX-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]


#ADABOOST
SA_AB = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\AdaBoost\\"+DATA_NAME+"\\SA-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]
CFX_AB = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\AdaBoost\\"+DATA_NAME+"\\CFX-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]


#SUPPORT VECTOR MACHINE
SA_SVM = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Support Vector Machine\\"+DATA_NAME+"\\SA-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]
CFX_SVM = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Support Vector Machine\\"+DATA_NAME+"\\CFX-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]


#KNN
SA_KNN = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\k-NN\\"+DATA_NAME+"\\SA-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]
CFX_KNN = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\k-NN\\"+DATA_NAME+"\\CFX-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]


#GAUSSIAN NAIVE BAYES
SA_GNB = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Gaussian Naive Bayes\\"+DATA_NAME+"\\SA-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]
CFX_GNB = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Gaussian Naive Bayes\\"+DATA_NAME+"\\CFX-TUNING\RESULTS\\F1SCORE_TEST.npy")[0]


#CHAOSNET
CFX_ = np.load("C:\\Users\\deeks\\OneDrive\\Documents\\Research\\Algorithms\\Chaosnet\\"+DATA_NAME+"\\CFX TUNING\RESULTS\\F1SCORE_TEST.npy")[0]



# creating the dataset
sa_data = {'DT':SA_DT, 'RF':SA_RF, 'AB':SA_AB,
        'SVM':SA_SVM, 'KNN':SA_KNN, 'GNB': SA_GNB}
cfx_data = {'ChaosNet':CFX_, 'CFX+DT':CFX_DT, 'CFX+RF':CFX_RF, 'CFX+AB':CFX_AB,
        'CFX+SVM':CFX_SVM, 'CFX+KNN':CFX_KNN, 'CFX+GNB': CFX_GNB}
courses = list(sa_data.keys())
values = list(sa_data.values())


courses_1 = list(cfx_data.keys())
values_1 = list(cfx_data.values())

fig = plt.figure(figsize = (15, 5))
plt.bar(courses_1[0], values_1[0], color ='green',
        width = 0.3)
# creating the bar plot
for x in range(0,len(courses)):
    plt.bar(courses[x], values[x], color ='white',edgecolor='black',linewidth='2.0',
        width = 0.3)
    plt.bar(courses_1[x+1], values_1[x+1], color ='blue',
        width = 0.3)
 
# plt.bar(courses, values, color ='white',edgecolor='black',linewidth='2.0',
#         width = 0.3)
# plt.bar(courses_1, values_1, color ='blue',
#         width = 0.3)
 

y_tix = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
plt.xticks(fontsize=16)
plt.yticks(y_tix, fontsize=25)
plt.xlabel("Algorithms", fontsize=25)
plt.ylabel("Macro-F1 Scores", fontsize=25)
# plt.title("Students enrolled in different courses")
plt.tight_layout()
plt.grid(axis='y', linestyle='-', linewidth=0.1)
plt.savefig(RESULT_PATH+DATA_NAME+"_BAR_PLOT.eps", format='eps', dpi=300)
plt.savefig(RESULT_PATH+DATA_NAME+"_BAR_PLOT.jpeg", format='jpeg', dpi=300)
plt.show()



