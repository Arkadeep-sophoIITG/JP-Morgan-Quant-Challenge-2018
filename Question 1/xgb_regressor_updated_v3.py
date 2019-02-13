from sklearn.metrics import log_loss,accuracy_score,roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import argparse
import os
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
import sys
import xgboost
from xgboost import XGBRegressor 
from sklearn.cross_validation import train_test_split
import time

def label_gender (row):
    if 'Dr' in row['Name']:
        return 4
    if 'Miss' in row['Name']:
        return 0
    if 'Mrs' in row['Name']:
        return 1
    if 'Mr' in row['Name']:
        return 3
    return None




def label_race (row):
   if (row['From'] == 1 and row['To'] == 2) or (row['From'] == 2 and row['To'] == 1):
        return 1446
   if (row['From'] == 1 and row['To'] == 3) or (row['From'] == 3 and row['To'] == 1):
        return 1654
   if (row['From'] == 1 and row['To'] == 4) or (row['From'] == 4 and row['To'] == 1):
        return 1148
   if (row['From'] == 1 and row['To'] == 5) or (row['From'] == 5 and row['To'] == 1):
        return 622
   if (row['From'] == 1 and row['To'] == 6) or (row['From'] == 6 and row['To'] == 1):
        return 1190
   if (row['From'] == 1 and row['To'] == 7) or (row['From'] == 7 and row['To'] == 1):
        return 1028
   if (row['From'] == 2 and row['To'] == 3) or (row['From'] == 3 and row['To'] == 2):
        return 470
   if (row['From'] == 2 and row['To'] == 4) or (row['From'] == 4 and row['To'] == 2):
        return 480
   if (row['From'] == 2 and row['To'] == 5) or (row['From'] == 5 and row['To'] == 2):
        return 1140
   if (row['From'] == 2 and row['To'] == 6) or (row['From'] == 6 and row['To'] == 2):
        return 437
   if (row['From'] == 2 and row['To'] == 7) or (row['From'] == 7 and row['To'] == 2):
        return 1485
   if (row['From'] == 3 and row['To'] == 4) or (row['From'] == 4 and row['To'] == 3):
        return 1307
   if (row['From'] == 3 and row['To'] == 5) or (row['From'] == 5 and row['To'] == 3):
        return 1180
   if (row['From'] == 3 and row['To'] == 6) or (row['From'] == 6 and row['To'] == 3):
        return 886
   if (row['From'] == 3 and row['To'] == 7) or (row['From'] == 7 and row['To'] == 2):
        return 1366
   if (row['From'] == 4 and row['To'] == 5) or (row['From'] == 5 and row['To'] == 4):
        return 1253
   if (row['From'] == 4 and row['To'] == 6) or (row['From'] == 6 and row['To'] ==  4):
        return 417
   if (row['From'] == 4 and row['To'] == 7) or (row['From'] == 7 and row['To'] == 4):
        return 1760
   if (row['From'] == 5 and row['To'] == 6) or (row['From'] == 6 and row['To'] == 5):
        return 1425
   if (row['From'] == 5 and row['To'] == 7) or (row['From'] == 7 and row['To'] == 5):
        return 520
   if (row['From'] == 6 and row['To'] == 7) or (row['From'] == 7 and row['To'] == 6):
        return 1534
   return None


def label_race (row):
    return time_from[str(int(row['From']))][int(row['To'])-1]

def dataframe_train(path):
    df = pd.read_csv(path,header=0,parse_dates=[1,4,6])
    train_Y = []
    train_Y = df['Fare']
    return df,train_Y

def dataframe_test(path):
    df = pd.read_csv(path,header=0,parse_dates=[1,4,6])
    return df


def objective(space):

    clf = XGBRegressor(n_estimators = space['n_estimators'],
                           max_depth = space['max_depth'],
                           min_child_weight = space['min_child_weight'],
                           subsample = space['subsample'],
                           learning_rate = space['learning_rate'],
                           gamma = space['gamma'],
                           colsample_bytree = space['colsample_bytree'],
                           reg_alpha= space['alpha'],
                           reg_lambda=space['lambda'],
                           booster= 'gbtree',                 
                           objective='reg:linear'
                           )

    eval_set  = [( train, y_train)]

    clf.fit(train,
            y_train,
            eval_set=eval_set,
            eval_metric = 'rmse')

    pred = clf.predict(train)
    rmse = mean_squared_error((y_train), (pred))

#    print "SCORE:", mae
    return{'loss':rmse, 'status': STATUS_OK }

def optimize(cores,random_state):
    space ={
            'max_depth': hp.choice('max_depth', np.arange(10, 70, dtype=int)),
            'min_child_weight': hp.quniform ('min_child_weight', 1, 20, 1),
            'subsample': hp.uniform ('subsample', 0.8, 1),
            'n_estimators' : hp.choice('n_estimators', np.arange(500, 5000, 100, dtype=int)),
            'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),
            'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'alpha' :  hp.quniform('alpha', 0, 10, 1),
            'lambda': hp.quniform('lambda', 1, 2, 0.1),
            'nthread': cores,
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'seed': random_state
        }


    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50, # change
                trials=trials)
    return best
    

# Loaded the entire training and test dataset into pandas dataframe
df_train,trainY = dataframe_train('train.csv')
df_test = dataframe_test('test.csv')

######  Computing new features based on training data    #######

# Added difference feature based on difference between Booking Date and Flight date
df_train['diff'] = df_train['Flight Date'] - df_train['Booking Date']
df_test['diff'] = df_test['Flight Date'] - df_test['Booking Date']
df_train['diff'] = df_train['diff'] / np.timedelta64(1, 'D')
df_test['diff'] = df_test['diff']/np.timedelta64(1,'D')


# Added feature age based on the birthdate of the passengers
df_train['age'] = df_train['Booking Date'] - df_train['Date of Birth']
df_train['age'] = df_train['age'] / np.timedelta64(1, 'D')
df_test['age'] = df_test['Booking Date'] - df_test['Date of Birth']
df_test['age'] = df_test['age'] / np.timedelta64(1, 'D')

# Added flight_month feature based on the month of the flight

df_train['flight_month'] = df_train['Flight Date'].dt.month
df_test['flight_month'] = df_test['Flight Date'].dt.month

# Added feature WEEKEND based on whether the flight is on weekday or weekend
df_train['WEEKEND'] = ((pd.DatetimeIndex(df_train['Flight Date']).dayofweek)//5+1).astype(float)
df_test['WEEKEND'] = ((pd.DatetimeIndex(df_test['Flight Date']).dayofweek)//5+1).astype(float)
#df_test['WEEKDAY'] = (pd.DatetimeIndex(df_test['Booking Date']).dayofweek).astype(float)
#df_train['WEEKDAY'] = (pd.DatetimeIndex(df_test['Booking Date']).dayofweek).astype(float)

# Added feature weekday based on flight date (Monday - 0 , Tuesday -1 , and so)
df_train['WEEKDAY'] = df_train['Flight Date'].dt.dayofweek
df_test['WEEKDAY'] = df_train['Flight Date'].dt.dayofweek

# Added feature week of flight based on flight date.

df_train['FLIGHTWEEK'] = df_train['Flight Date'].dt.week
df_test['FLIGHTWEEK'] = df_test['Flight Date'].dt.week

df_train = df_train.drop(columns=['Flight Date', 'Booking Date'])
df_test = df_test.drop(columns=['Flight Date', 'Booking Date'])
df_train = df_train.drop(columns='Date of Birth')
df_test = df_test.drop(columns='Date of Birth')


dict_cities = {'Mumbai':1 , 'Patna' : 2, 'Kolkata' : 3, 'Delhi' : 4, 'Hyderabad': 5, 'Lucknow': 6, 'Chennai':7}
dict_class = {'Business':25, 'Economy': 10}
df_train = df_train.replace({'From':dict_cities})
df_test = df_test.replace({'From':dict_cities})
df_train = df_train.replace({'To': dict_cities})
df_test = df_test.replace({'To': dict_cities})
df_train = df_train.replace({'Class':dict_class})
df_test = df_test.replace({'Class':dict_class})

df_train['Flight Time'] = df_train['Flight Time'].str.split(':').str[0]
df_train['age'] = df_train['age']/365.0
df_test['Flight Time'] = df_test['Flight Time'].str.split(':').str[0]
df_test['age'] = df_test['age']/365.0

time_from = {}
time_from['1'] = [0,130,140,120,75,130,89]
time_from['2'] = [205,0,60,95,104,39,134]
time_from['3'] = [155,90,0,120,120,105,135]
time_from['4'] = [115,80,115,0,120,55,155]
time_from['5'] = [70,104,115,125,0,235,65]
time_from['6'] = [130,39,100,60,245,0,138]
time_from['7'] = [95,134,115,155,65,138,0]



df_train['flight_duration'] = df_train.apply (lambda row: label_race (row),axis=1)
df_test['flight_duration'] = df_test.apply (lambda row: label_race (row),axis=1)
df_train['Flight Time'] = df_train['Flight Time'].astype(str).astype(int)
df_test['Flight Time'] = df_test['Flight Time'].astype(str).astype(int)
df_train['Name'] = df_train['Name'].str.split('.').str[0]
df_test['Name'] = df_test['Name'].str.split('.').str[0]

df_train['gender'] = df_train.apply (lambda row: label_gender (row),axis=1)
df_test['gender'] = df_test.apply (lambda row: label_gender (row),axis=1)
df_train= df_train.drop(columns='Name')
df_test = df_test.drop(columns='Name')
df_train =df_train.drop(columns='Fare')
train = df_train
y_train = trainY


best = optimize(24,1234)
print(best)
xgb = XGBRegressor(n_estimators = int(best['n_estimators']),
                    learning_rate= best['learning_rate'],objective="reg:linear",booster='gbtree',
                    gamma =  best['gamma'],max_depth=best['max_depth'],
                    min_child_weight=int(best['min_child_weight']),subsample=best['subsample'],
                    colsample_bytree=best['colsample_bytree'],reg_alpha=best['alpha'],
                    reg_lambda=best['lambda'],nthread=24,random_state=1234)


xgb.fit(df_train,trainY)
print(xgb.score(df_train,trainY))


xgb_final_preds = xgb.predict(df_test)
np.savetxt('/home/arkadeep/scratch/finall_please_finallll_donee.csv',xgb_final_preds,delimiter = ',')