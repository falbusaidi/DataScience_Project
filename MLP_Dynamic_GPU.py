# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:43:53 2018

@author: fahad
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:07:59 2018

@author: fahad
@purpose : a MLP to predict RUL for the C-mapss dataset using a dynamic MLP. This cope is deisgned to run for GPU

"""

# import required libraries
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
import math
from prepare_training_dataset import load_Training_data
from prepare_test_dataset import load_test_data
from keras.callbacks import EarlyStopping

## Global variable
#file = int(sys.argv[1])
#unit = int(sys.argv[2])
file = 1
unit=0
layers   = 2

# calculate RUL score
def calScore(test, pred):
    result = 0 
    for i in range (0,len(test)):
        d =  pred[i] - test[i]
        score =0
        if d < 0 :
            score = (math.exp(-d/10)) - 1
            result = result + score
            
        elif d > 0  :
            score = (math.exp(d/13)) - 1
            result = result + score
    return result

# Build MLP the Model
def buildMLP(layers , inputnum):
    
    layer_1 = int((inputnum+1)*(2.0/3.0))
    layer2  = int((inputnum+layer_1+1)*(2.0/3.0))
    model = Sequential()
    model.add(Dense(units= layer_1, kernel_initializer = 'uniform',  activation = 'relu', input_dim = inputnum))
    model.add(Dense(units= layer2, kernel_initializer = 'uniform' , activation = 'relu'))
    model.add(Dense(units= 1, kernel_initializer = 'uniform'))

    # compile the model
    model.compile (loss ='mean_squared_error', optimizer ='adam',metrics= ['mse'])         
    return model, layer_1, layer2

# load data files

df_train = load_Training_data(file)
df_test = load_test_data(file) 



# filter dataframe by ID
if unit != 0:
    df_train= df_train[df_train['UNIT_ID']==unit]
    df_test = df_test[df_test['UNIT_ID']==unit]
    print('result filtered for ID',unit)

# create the training and testing sets from the dataframes
training_set = df_train.iloc[:,2:].values
test_set = df_test.iloc[:,2:].values



# scaling 
scaler = MinMaxScaler((-1,1))
training_scaled = scaler.fit_transform(training_set)
test_scaled = scaler.transform(test_set)


# create the scaled train and test sets
X_train = training_scaled[0:,0:24]
y_train = training_scaled[:,24]

X_test = test_scaled[0:,0:24]
y_test = test_scaled[:,24]

# scale for inversing prediction values based on test set target
sc_predict = MinMaxScaler(feature_range=(-1,1))
sc_predict.fit_transform(test_set[:,24:25])

# feature extraction
#col = [3,7,8,12,18,20,21]
#col = [4,8,9,13,19,21,22]
#individual = [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
#col = [index for index in range(len(individual)) if individual[index] == 0]
#X_train = np.delete(X_train,col,1)
#X_test =  np.delete(X_test,col,1)

#X_train = np.delete(X_train,col,1)
#X_test =  np.delete(X_test,col,1)


# build and initialize the model 

model, layer_1,layer_2    = buildMLP(layers,X_train.shape[1])

# early stopping 
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=False)

# Fitting the ANN to the Training set
start = time.time()
history = model.fit(X_train, y_train, batch_size = 64, epochs = 1000, shuffle=False, verbose=False, callbacks=[es])
end = time.time()

# display the results

print('=========Parameters==============')
print('File FD00',file)
print('number of nodes in layer 1:',layer_1)
print('number of nodes in layer 2:',layer_2)
print ('filtered ID: ',unit)

print('=========Results==============')
list = history.history['mean_squared_error']
np.sqrt(np.array(list).mean())
print('trainig perfromance is MSE:',np.array(list).mean())
print('Training perfromance is RMSe:',np.sqrt(np.array(list).mean()))


# predicting based on test data
prediction = model.predict(X_test)
prediction  = sc_predict.inverse_transform(prediction)
y_test = sc_predict.inverse_transform(y_test.reshape(-1,1))

print('Testing perfromance is MSE:',mean_squared_error(y_test,prediction))
print('Testing perfromance is RMSe:',np.sqrt(mean_squared_error(y_test,prediction)))
print('Model runtime is %0.2f seconds'%(end-start))
print(calScore(y_test,prediction))   
model.summary()


# plotting    
#
#fig, axs = plt.subplots(1,4,figsize=(15,5))
#
#index = df_test[df_test['UNIT_ID']==21].index
#axs[0].plot(df_test.iloc[index[0]:(index[-1]+1),1],y_test[index[0]:(index[-1]+1)])
#axs[0].plot(df_test.iloc[index[0]:(index[-1]+1),1],prediction[index[0]:(index[-1]+1)])
#axs[0].set_title('Sequence for Unit 21')
#axs[0].set_xlabel('Cycle')
#axs[0].set_ylabel('RUL')
#
#
#index = df_test[df_test['UNIT_ID']==24].index
#axs[1].plot(df_test.iloc[index[0]:(index[-1]+1),1],y_test[index[0]:(index[-1]+1)])
#axs[1].plot(df_test.iloc[index[0]:(index[-1]+1),1],prediction[index[0]:(index[-1]+1)])
#axs[1].set_title('Sequence for Unit 24')
#axs[1].set_xlabel('Cycle')
#axs[1].set_ylabel('RUL')
#
#
#
#index = df_test[df_test['UNIT_ID']==34].index
#axs[2].plot(df_test.iloc[index[0]:(index[-1]+1),1],y_test[index[0]:(index[-1]+1)])
#axs[2].plot(df_test.iloc[index[0]:(index[-1]+1),1],prediction[index[0]:(index[-1]+1)])
#axs[2].set_title('Sequence for Unit 34')
#axs[2].set_xlabel('Cycle')
#axs[2].set_ylabel('RUL')
#
#index = df_test[df_test['UNIT_ID']==100].index
#axs[3].plot(df_test.iloc[index[0]:(index[-1]+1),1],y_test[index[0]:(index[-1]+1)])
#axs[3].plot(df_test.iloc[index[0]:(index[-1]+1),1],prediction[index[0]:(index[-1]+1)])
#axs[3].set_title('Sequence for Unit 100')
#axs[3].set_xlabel('Cycle')
#axs[3].set_ylabel('RUL')
#
#plt.show()





    
  

