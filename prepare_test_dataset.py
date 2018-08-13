
"""
Created on Thu Jun  7 13:07:59 2018

@author: fahad
@purpose : The purpose of this code is to prepare the test dataset and calculate the RUL. 
The program will read the required C-MAPSS dataset file and will return the dataset with calculate RUL as 
a dataframe. Also, will save the dataframe to a CSV file 
"""
# import the required packages 
import numpy as np
import pandas as pd





# define file column heading
col_header = np.array(['UNIT_ID','CYCLE_ID','CONDITION_1','CONDITION_2','CONDITION_3','SENSOR_01','SENSOR_02','SENSOR_03','SENSOR_04',
                   'SENSOR_05','SENSOR_06','SENSOR_07','SENSOR_08','SENSOR_09','SENSOR_10','SENSOR_11','SENSOR_12','SENSOR_13',
                   'SENSOR_14','SENSOR_15','SENSOR_16','SENSOR_17','SENSOR_18','SENSOR_19','SENSOR_20','SENSOR_21'])





# funtion to calculate single RUL
def calcRUL(cycle,max_cycles,rul):
    cal_rul = rul+max_cycles-cycle
    #if cycle == max_cycles:
    #    cal_rul = rul 
    if  cal_rul >= 119:
        cal_rul = 119
    return cal_rul 

# function to calculate Rul for a file 
def populateRUL(df_test,df_rul):
    result = pd.DataFrame()
    df_ids = df_test['UNIT_ID'].unique()
    for ids in df_ids:
        df_test1 = df_test[df_test['UNIT_ID'] == ids].copy()
        df_test1['RUL']=-1
        cycles = df_test1['CYCLE_ID'].max()
        index = ids -1
        rul = df_rul.iloc[ids -1]['RUL']
        df_test1['RUL'] = df_test1['CYCLE_ID'].apply(lambda x: calcRUL(x,cycles,rul))
        result = result.append(df_test1)
    return result





# populate traing file with RUl column and save results to csv file and create all traing dataframe


def load_test_data(number):
    df_test = pd.read_csv('CMAPSSData/test_FD00'+str(number)+'.txt', delim_whitespace=True, header=None, names=col_header)
    df_rul  = pd.read_csv('CMAPSSData/RUL_FD00'+str(number)+'.txt', delim_whitespace=True, header=None, names=['RUL'])
    df_test = populateRUL(df_test,df_rul)
    df_test.to_csv('inputData/test_FD00'+str(number)+'_kincked.csv',mode='w', index=False)
    return df_test

