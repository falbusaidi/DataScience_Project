
# coding: utf-8


"""
Created on Thu Jun  7 13:07:59 2018

@author: fahad
@purpose : The purpose of this code is to prepare the training dataset and calculate the RUL. 
The program will read the required C-MAPSS dataset file and will return the dataset with calculate RUL as 
a dataframe. Also, will save the dataframe to a CSV file 
"""
# import packages 
import numpy as np
import pandas as pd




# define file column heading
col_header = np.array(['UNIT_ID','CYCLE_ID','CONDITION_1','CONDITION_2','CONDITION_3','SENSOR_01','SENSOR_02','SENSOR_03','SENSOR_04',
                   'SENSOR_05','SENSOR_06','SENSOR_07','SENSOR_08','SENSOR_09','SENSOR_10','SENSOR_11','SENSOR_12','SENSOR_13',
                   'SENSOR_14','SENSOR_15','SENSOR_16','SENSOR_17','SENSOR_18','SENSOR_19','SENSOR_20','SENSOR_21'])




# funtion to calculate single RUL
def calcRUL(x,y):
    rul = y-x
    if rul >= 119:
        rul = 119
    return rul 

# function to calculate Rul for a file 
def populateRUL(data1):
    data1['RUL']=-1
    result = pd.DataFrame()
    df_ids = data1['UNIT_ID'].unique()
    for ids in df_ids:
        df_data1 = data1[data1['UNIT_ID'] == ids].copy()
        maxc = df_data1['CYCLE_ID'].max()
        df_data1['RUL'] = df_data1[df_data1['UNIT_ID'] == ids]['CYCLE_ID'].apply(lambda x: calcRUL(x,maxc))
        result = result.append(df_data1)
    return result



# populate traing file with RUl column and save results to csv file and create all traing dataframe


def load_Training_data(number):
    df = pd.read_csv('CMAPSSData/train_FD00'+str(number)+'.txt', delim_whitespace=True, header=None, names=col_header)
    df = populateRUL(df)
    df.to_csv('inputData/Train_FD00'+str(number)+'.csv',mode='w', index=False)
    return df



