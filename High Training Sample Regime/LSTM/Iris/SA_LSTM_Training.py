"""
Created on Mon Feb 20, 2023

Author:  Shubham Raheja (shubhamraheja1999@gmail.com)
         Deeksha Sethi  (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of stand-alone LSTM on the Iris dataset.
Dataset Source: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#

"""

import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras_tuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping
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
   
_______________________________________________________________________________

DL hyperparameter description:
_______________________________________________________________________________

    LSTM units - Positive integer, dimensionality of the output space. 
    Source : Keras 
    (https://keras.io/api/layers/recurrent_layers/lstm/)
    
    Dense layer units - Positive integer, dimensionality of the output space.
    Source : Keras 
    (https://keras.io/api/layers/core_layers/dense/)
    
    Dense layer activation - Activation function of the dense layer.
    Source : Keras
    (https://keras.io/api/layers/core_layers/dense/)
    
    Dropout rate - Float between 0 and 1. Fraction of the input units to drop.
    The dropout layer randomly sets input units to 0 with a frequency of rate 
    at each step during training time, which prevents overfitting. 
    Source : Keras
    (https://keras.io/api/layers/regularization_layers/dropout/)
    
    Learning rate - The learning rate is a hyperparameter that controls how 
    much to change the model in response to the estimated error each time the 
    odel weights are updated.
    
    Here the learning rate is being set for the Adam optimizer. 
    
    Adam optimization is a stochastic gradient descent method 
    that is based on adaptive estimation of first-order and second-order 
    moments.
    Source : Keras
    (https://keras.io/api/optimizers/adam/)
    
    
    The above mentioned hyperparameters are tuned using KerasTuner. It is a 
    general-purpose hyperparameter tuning library.
    Source : Keras
    (https://keras.io/guides/keras_tuner/)
    
_______________________________________________________________________________

Performance metric used:
_______________________________________________________________________________

    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and finds  their 
    unweighted mean. This does not take label imbalance into account.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
_______________________________________________________________________________

'''



# Import the IRIS Dataset from sklearn library
iris = datasets.load_iris()

# Reading data and labels from the dataset
X = iris.data
y = iris.target

# Normalisation of the data attributes - Column-wise
X = (X - np.min(X,0))/(np.max(X,0) - np.min(X,0))
X = np.reshape(X,(X.shape[0], 1, X.shape[1])) # Reshaping as tensor for LSTM algorithm.

# Binary matrix representation of the labels
y = to_categorical(y)

# Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Algorithm - LSTM / Building the model

def model_builder(hp):
    model = Sequential()
    hp_units = hp.Int('units',min_value=8,max_value=128,step=8) # Selecting the number of LSTM units; min units = 8, max units = 128, step size = 8
    hp_dense = hp.Int('dense',min_value=8,max_value=128,step=8) # Selecting the number of dense units; min units = 8, max units = 128, step size = 8                                                                                                                        
    hp_activation = hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu') # Selecting the activation for dense layer
    hp_dropout_rate = hp.Float('dropout_rate',min_value=0,max_value=0.5,step=0.1) # Selecting the dropout rate
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) # Selecting the learning rate
                              
    model.add(LSTM(units=hp_units, input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(Dropout(hp_dropout_rate))
    model.add(Dense(units=hp_dense,activation=hp_activation))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=hp_learning_rate),
                  metrics = ['accuracy'])
    return model


# Defining a Tuner class to run the search
tuner= RandomSearch(
        model_builder, # Model-building function 
        objective='val_accuracy', # Objective to optimize
        max_trials=50, # Total number of trials to run during the search
        overwrite=True, # Overwrite the previous results in the same directory or resume from the previous search
        directory = 'SA-TUNING', # A path to a directory for storing the search results
        project_name = 'TRIALS', # Name of the sub-directory in the directory (SA-TUNING)
        executions_per_trial=1 # Number of models to be built and fit for each trial
        )

# Stop early if validation loss remains the same for 3 epochs
stop_early = EarlyStopping(monitor='val_loss',patience = 3)

# Start the search    
tuner.search(X_train,y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks = [stop_early]
)

# Best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
units = best_hps.get('units')
dense = best_hps.get('dense')
dense_activation = best_hps.get('dense_activation')
dropout_rate = best_hps.get('dropout_rate')
learning_rate = best_hps.get('learning_rate')


# Re-build the LSTM model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train,
                    y_train,
                    epochs = 50,
                    validation_split = 0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

# Printing best hyperparameters
print('Best Hyperparameters:')
print('LSTM Units:', units)
print('Dense Layer Units:', dense)
print('Dense Layer Activation Function:', dense_activation)
print('Dropout Rate:', dropout_rate)
print('Learning Rate:', learning_rate)
print('Best number of epochs:', best_epoch)

print("Saving Hyperparameter Tuning Results")

PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/'

try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

np.save(RESULT_PATH+"/h_Units.npy", (units)) 
np.save(RESULT_PATH+"/h_Dense.npy", (dense)) 
with open(RESULT_PATH+"/h_Activation.txt",'w') as file:
    file.write(dense_activation)
np.save(RESULT_PATH+"/h_DropoutRate.npy", (dropout_rate)) 
np.save(RESULT_PATH+"/h_LearningRate.npy", (learning_rate)) 
np.save(RESULT_PATH+"/h_BestEpoch.npy", best_epoch) 


