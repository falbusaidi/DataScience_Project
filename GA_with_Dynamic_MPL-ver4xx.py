
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:08:16 2018

@author: fahad
@purpose:  the purpose of this code is to use a genetic algorithm to determine best featuere subset for a dynamic ANN and the number of 
neurons in each of ANN layers. 
The code utilises the Keras and DEAP library for GA . 
This code is based on GA code example by Stuart Colianni on Github ttps://github.com/scoliann/GeneticAlgorithmFeatureSelection

"""

# import packages
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from deap import base
from deap import creator
from deap import tools
import random
from deap import algorithms
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
from prepare_training_dataset import load_Training_data
from prepare_test_dataset import load_test_data
import sys
import multiprocessing
import timeit


# define the global variables provided by the command line
script = sys.argv[0]
file = int(sys.argv[1])
unit = int(sys.argv[2])
numPop = int(sys.argv[3])
numGen = int(sys.argv[4])

# load the training and testing dataset
df_train = load_Training_data(file)
df_test = load_test_data(file)

# filter dataset
if unit != 0:
    df_train= df_train[df_train['UNIT_ID']==unit]
    df_test = df_test[df_test['UNIT_ID']==unit]


training_set = df_train.iloc[:,2:].values
test_set = df_test.iloc[:,2:].values

# scaling 

scaler = MinMaxScaler((-1,1))
training_scaled = scaler.fit_transform(training_set)
test_scaled = scaler.transform(test_set)



# create the train and test sets
X_train = pd.DataFrame(training_scaled[0:,0:24])
y_train = training_scaled[:,24]

X_test = pd.DataFrame(test_scaled[0:,0:24])
y_test = test_scaled[:,24]

# scale for inversing prediction values based on test targets
sc_predict = MinMaxScaler(feature_range=(-1,1))
sc_predict.fit_transform(test_set[:,24:25])






# build the model 

def build_model(inputnum):
    
    layer_1 = int((inputnum+1)*(2.0/3.0))
    layer_2 = int((inputnum+layer_1+1)*(2.0/3.0))   
    model = Sequential()
    model.add(Dense(units= layer_1, kernel_initializer = 'uniform', activation = 'relu', input_dim = inputnum))
    model.add(Dense(units= layer_2, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units= 1, kernel_initializer = 'uniform'))
    #optimizer
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # compile the model
    model.compile (
                loss ='mean_squared_error',
                optimizer ='adam',
                metrics= ['mse']
    )
    return model

# the fitness function trains the model and use the RMSE on test data as the individual fitness

def getFitness(individual, X_train, X_test, y_train, y_test, scale):
    # Parse our feature columns that we don't use
    # Apply one hot encoding to the features
    cols = [index+2 for index in range(len(individual)) if individual[index] == 0]
    X_trainParsed = X_train.drop(X_train.columns[cols], axis=1)
    X_testParsed = X_test.drop(X_test.columns[cols], axis=1)
    col_num = len(X_trainParsed.columns)
    X_trainParsed = X_trainParsed.values
    X_testParsed = X_testParsed.values
    # Apply logistic regression on the data, and calculate accuracy
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=False)
    clf = build_model(col_num)
    clf.fit(X_trainParsed, y_train, batch_size = 1000, epochs = 64, verbose=False , callbacks=[es], validation_split=0.2)
    predictions = clf.predict(X_testParsed)
    predictions  = scale.inverse_transform(predictions)
    actual_values = scale.inverse_transform(y_test.reshape(-1, 1))
    accuracy = mean_squared_error(actual_values, predictions)

    # Return calculated accuracy as fitness
    return (accuracy,)


# initialising Deap variables 

# initialise Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
# Create Toolbox and define the mechanism to initialise individuals and population
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X_train.columns)-3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# fill toolbox evaluation paramater, crossover type, mutation function, and selection method
toolbox.register("evaluate", getFitness, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, scale=sc_predict)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)



# assigning the mutation and corssover probabilities
crx_prob= 0.6
mu_prob=0.001 

pop = toolbox.population(n=numPop)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)






if __name__ == '__main__':
#    pool = multiprocessing.Pool()
#    toolbox.register("map", pool.map)
    # run the genetic algorithm
    start = timeit.default_timer()
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crx_prob, mutpb=mu_prob, ngen=numGen, stats=stats, halloffame=hof, verbose=True)
    stop = timeit.default_timer()
    
    # print the results
    print (stop - start)
    print('====Programe starting with parameters=====')
    print('the script is :', script)
    print('input file FD00',file)
    print('result filtered for ID',unit)
    print('population:',numPop)
    print('generation num:',numGen)
    print('crossover prob:', crx_prob)
    print('mutation', mu_prob)
    print('===============Results====================')
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
       
        
        
        
            
        
    
