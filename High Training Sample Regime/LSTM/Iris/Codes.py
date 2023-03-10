# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Date: Tue 22 Dec, 2020


Code Description : ChaosNet decision function for CFX+LSTM


Updated for LSTM on Fri 24 Feb, 2023 by Shubham Raheja (shubhamraheja1999@gmail.com)

"""

import os
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score
import ChaosFEX.feature_extractor as CFX
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras_tuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping


def k_cross_validation(X_train, y_train, X_test, y_test, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON):
    """

    Parameters
    ----------
    X_train : TYPE-numpy 2D array
        DESCRIPTION - Traindata
    y_train : TYPE-numpy 2D array
        DESCRIPTION - Trainlabel
    X_test : TYPE-numpy 2D array
        DESCRIPTION - Testdata
    y_test : TYPE - numpy 2D array
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
    y_train = to_categorical(y_train)
    BESTF1 = 0
    FSCORE_TEMP=[]
    for DT in DISCRIMINATION_THRESHOLD:
        
        for INA in INITIAL_NEURAL_ACTIVITY:
            
            for EPSILON_1 in EPSILON:
   
                # Extract features
                X_train_cfx = CFX.transform(X_train, INA, 10000, EPSILON_1, DT)
                X_test_cfx = CFX.transform(X_test, INA, 10000, EPSILON_1, DT)
                # Reshaping as tensor for LSTM algorithm.            
                X_train_cfx = np.reshape(X_train_cfx,(X_train_cfx.shape[0], 1, X_train_cfx.shape[1]))
                X_test_cfx = np.reshape(X_test_cfx,(X_test_cfx.shape[0], 1, X_test_cfx.shape[1]))
                
                def model_builder(hp):
                     model = Sequential()
                     hp_units = hp.Int('units',min_value=8,max_value=128,step=8) # Selecting the number of LSTM units; min units = 8, max units = 128, step size = 8
                     hp_dense = hp.Int('dense',min_value=8,max_value=128,step=8) # Selecting the number of dense units; min units = 8, max units = 128, step size = 8                                                                                                                        
                     hp_activation = hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu') # Selecting the activation for dense layer
                     hp_dropout_rate = hp.Float('dropout_rate',min_value=0,max_value=0.5,step=0.1) # Selecting the dropout rate
                     hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) # Selecting the learning rate
                                                          
                     model.add(LSTM(units=hp_units, input_shape=(X_train_cfx.shape[1],X_train_cfx.shape[2])))
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
                     objective='accuracy', # Objective to optimize
                     max_trials=50, # Total number of trials to run during the search
                     overwrite=True, # Overwrite the previous results in the same directory or resume from the previous search
                     directory = 'CFX-TUNING', # A path to a directory for storing the search results
                     project_name = 'TRIALS', # Name of the sub-directory in the directory (SA-TUNING)
                     executions_per_trial=1 # Number of models to be built and fit for each trial
                     )
                            
                # Stop early if validation loss remains the same for 3 epochs
                stop_early = EarlyStopping(monitor='val_loss',patience = 3)
                 
                # Start the search    
                tuner.search(X_train_cfx,
                              y_train,
                              epochs=50,
                              batch_size=32,
                              validation_split=0.2,
                              callbacks = [stop_early]
                            )
                # Best Hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                            
                # Re-build the LSTM model with the best hyperparameters
                model = tuner.hypermodel.build(best_hps)
                history = model.fit(X_train_cfx,
                                     y_train,
                                     epochs = 50,
                                     validation_split = 0.2)
                            
                val_acc_per_epoch = history.history['val_accuracy']
                best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
                 
                hypermodel = tuner.hypermodel.build(best_hps)
                hypermodel.fit(X_train_cfx,
                                y_train,
                                epochs = best_epoch)
                # Make predictions with trained model
                y_pred_testdata = np.argmax(hypermodel.predict(X_test_cfx), axis=-1)
                #y_test= np.argmax(y_test,axis=1)
                F1SCORE = f1_score(y_test, y_pred_testdata, average="macro")
                print("F1-Score for Q = ", INA,"B = ", DT,"EPSILON = ", EPSILON_1," is  = ", (F1SCORE)) 
                FSCORE_TEMP.append(F1SCORE)
                
                if(F1SCORE > BESTF1):
                    BESTF1 = F1SCORE
                    BESTINA = INA
                    BESTDT = DT
                    BESTEPS = EPSILON_1
                    BEST_units = best_hps.get('units')
                    BEST_dense = best_hps.get('dense')
                    BEST_dense_activation = best_hps.get('dense_activation')
                    BEST_dropout_rate = best_hps.get('dropout_rate')
                    BEST_learning_rate = best_hps.get('learning_rate')
            

    print(FSCORE_TEMP)
    
    # Save the obtained hyperparameters
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
    np.save(RESULT_PATH+"/h_Units.npy", (BEST_units)) 
    np.save(RESULT_PATH+"/h_Dense.npy", (BEST_dense)) 
    with open(RESULT_PATH+"/h_Activation.txt",'w') as file:
        file.write(BEST_dense_activation)
    np.save(RESULT_PATH+"/h_DropoutRate.npy", (BEST_dropout_rate)) 
    np.save(RESULT_PATH+"/h_LearningRate.npy", (BEST_learning_rate)) 
    np.save(RESULT_PATH+"/h_BestEpoch.npy", best_epoch) 
    np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) ) 

    # Print the saved hyperparameters
    print("BEST F1SCORE", BESTF1)
    print("BEST INITIAL NEURAL ACTIVITY = ", BESTINA)
    print("BEST DISCRIMINATION THRESHOLD = ", BESTDT)
    print("BEST EPSILON = ", BESTEPS)
    print('LSTM Units:', BEST_units)
    print('Dense Layer Units:', BEST_dense)
    print('Dense Layer Activation Function:', BEST_dense_activation)
    print('Dropout Rate:', BEST_dropout_rate)
    print('Learning Rate:', BEST_learning_rate)
    print('Best number of epochs:', best_epoch)


    





