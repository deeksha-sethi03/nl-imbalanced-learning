"""
Created on Thu Dec 2, 2021

Author: Deeksha Sethi (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of stand-alone Random Forest on the Iris dataset.
Dataset Source: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#

"""
import os
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    Iris Setosa       -     0
    Iris Versicolor   -     1
    Iris Virginica    -     2
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________

    iris            -   Complete Iris dataset.    
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


#import the IRIS Dataset from sklearn library
iris = datasets.load_iris()

#reading data and labels from the dataset
X = iris.data
y = iris.target


#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


#Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0))/(np.max(X_train,0) - np.min(X_train,0))
X_test_norm = (X_test - np.min(X_test,0))/(np.max(X_test,0) - np.min(X_test,0))


#Algorithm - Random Forest
n_estimator = [1, 10, 100, 1000, 10000]
BESTF1 = 0
FOLD_NO = 5
KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
KF.get_n_splits(X_train_norm) 
print(KF) 
for NEST in n_estimator:
                
    for MD in range(1,11):


        FSCORE_TEMP=[]
    
        for TRAIN_INDEX, VAL_INDEX in KF.split(X_train_norm):
            
            X_TRAIN, X_VAL = X_train_norm[TRAIN_INDEX], X_train_norm[VAL_INDEX]
            Y_TRAIN, Y_VAL = y_train[TRAIN_INDEX], y_train[VAL_INDEX]
        
            
            clf = RandomForestClassifier( n_estimators = NEST, max_depth = MD, random_state=42)
            clf.fit(X_TRAIN, Y_TRAIN.ravel())
            Y_PRED = clf.predict(X_VAL)
            f1 = f1_score(Y_VAL, Y_PRED, average='macro')
            FSCORE_TEMP.append(f1)
            print('F1 Score', f1)
        print("Mean F1-Score for N-EST = ", NEST," MD = ", MD," is  = ",  np.mean(FSCORE_TEMP)  )
        if(np.mean(FSCORE_TEMP) > BESTF1):
            BESTF1 = np.mean(FSCORE_TEMP)
            BESTNEST = NEST
            BESTMD = MD
            

print("BEST F1SCORE", BESTF1)
print("BEST MD = ", BESTMD)
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
np.save(RESULT_PATH+"/h_MD.npy", np.array([BESTMD]) ) 
np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) ) 

