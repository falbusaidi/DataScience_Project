# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:58:51 2018

@author: fahad
purpose: LSTM model to predict RUL using the C-MAPSS datase, this code is designed to run on GPU 

Reference: This code template was taken from the RNN code exmaple by Kirill Eremenko and Hadelin de Ponteves
part of Udemy.com online course "Deep Learning A-Z™: Hands-On Artificial Neural Networks"
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import sys
#import matplotlib.pyplot as plt
import pandas as pd
from prepare_training_dataset import load_Training_data
from prepare_test_dataset import load_test_data
from sklearn.metrics import mean_squared_error
import math, time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler

# Global variable
file = int(sys.argv[1]) # representing the file number ( fd001, fd002,003,004)
unit = int(sys.argv[2]) # Unit ID for filtering the dataset
step_size = int(sys.argv[3])

# scoring function
def calScore(test, pred):
    result = 0 
    for i in range (len(test)):
        d =  pred[i] - test[i]
        score =0        
        if d < 0 :
            score = (math.exp(- (d/10))) - 1
            result = result + score
            
        elif d > 0  :
            score = (math.exp(d/13)) - 1
            result = result + score
    return result




# Importing the training dataframe

dataset_train = load_Training_data(file)

# filter dataframe by ID
if unit != 0:
    dataset_train = dataset_train[dataset_train['UNIT_ID']==unit]

# construct the dataset as numpy array

training_set = dataset_train.iloc[:,2:].values

# Feature Scaling


sc = MinMaxScaler(feature_range = (-1, 1))
training_set_scaled = sc.fit_transform(training_set[:, 0:])



#restore original values of RUL
#training_set_scaled[:,24]=dataset_train.iloc[:,26].values

# Creating a data structure with 60 timesteps and 1 output

X_train = []
y_train = []
for i in range(step_size, len(dataset_train)):
    X_train.append(training_set_scaled[i-step_size:i,0:])
    y_train.append(training_set_scaled[i,24])
    
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 25))



# Initialising the LSTM
regressor = Sequential()

# Adding the first LSTM layeR
regressor.add(CuDNNLSTM(units = 200, return_sequences = True, input_shape = (X_train.shape[1], 25)))


# Adding a second LSTM layer 
regressor.add(CuDNNLSTM(units = 200, return_sequences = True))

# Adding a THE LAST LSTM layer 
regressor.add(CuDNNLSTM(units = 100))


# Adding the output layer
regressor.add(Dense(units = 1))

RMSprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1)

# Fitting the LSTM to the Training set
start = time.time()
regressor.fit(X_train, y_train, epochs = 99, batch_size = 32, callbacks=[es], validation_split=0.2)
end = time.time()

#loading the test data
dataset_test = load_test_data(file)

# filter dataset by ID
if unit != 0:
    dataset_test = dataset_test[dataset_test['UNIT_ID']==unit]

# select feature from the dataframe
test_set = dataset_test.iloc[:,2:].values

# scale the test data
test_sc = MinMaxScaler(feature_range = (-1, 1))
inputs  = test_sc.fit_transform(test_set)


# scale for the predicted values
sc_predict = MinMaxScaler(feature_range=(-1,1))
sc_predict.fit_transform(test_set[:,24:25])

X_test = []
y_test = []
for i in range(step_size, len(inputs)):
    X_test.append(inputs[i-step_size:i,0:])
    y_test.append(inputs[i,24])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 25))

# predict using test set

prediction = regressor.predict(X_test)

# re-inverse the values for the prediction and actual values 
prediction = sc_predict.inverse_transform(prediction)
actual_values = sc_predict.inverse_transform(y_test.reshape(-1,1))

# print the results

print('=========Parameters==============')
print('File FD00',file)
print('Time steps:',step_size)
print ('filtered ID: ',unit)
print('=========Results==============')
print('Testing perfromance is MSE:',mean_squared_error(actual_values,prediction))
print('Testing perfromance is RMSe:',np.sqrt(mean_squared_error(actual_values,prediction)))
print('Model runtime is %0.2f seconds'%(end-start))
print('score is',calScore(actual_values,prediction))   

# saving the files as result arrays python pickel 
actual_file = "model_results/LSTM_ACTUAL_FD00"+str(file)+".npy"
pred_file = "model_results/LSTM_PREDICTED_FD00"+str(file)+".npy"
np.save(actual_file, actual_values)
np.save(pred_file, prediction)
## Visualising the results
#plt.plot(actual_values, color = 'red', label = 'Actual')
#plt.plot(prediction, color = 'blue', label = 'predicted')
#plt.title('RUL Prediction')
#plt.xlabel('Time')
#plt.ylabel('RUL')
#plt.legend()
#plt.show()



