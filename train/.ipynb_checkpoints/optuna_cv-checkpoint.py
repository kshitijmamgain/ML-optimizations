import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV

import csv
from timeit import default_timer as timer
import random

import optuna.integration.lightgbm as lgbo
import optuna

# starting with storing the data as data frame
df = pd.read_csv("/home/jupyter/train_test_files/sample.csv")
df.drop( columns='Unnamed: 0', inplace =True)

# making a smaller df for quick testing
df_s, _ = train_test_split(df, random_state = 30, train_size = 0.01)

train_X = df_s.drop(columns = '0')
train_y = df_s['0']

train_set = lgb.Dataset(data=train_X, label = train_y)

MAX_EVALS = 5
N_FOLDS = 10

def objective(trial):
    
    dtrain = lgbo.Dataset(train_X, label=train_y)

    global ITERATION_O

    ITERATION_O += 1

    param ={              
            'num_leaves': trial.suggest_int('num_leaves', 16, 196, 4),

            'max_bin' : trial.suggest_uniform('max_bin', 254, 254), #if using CPU just set this to 254

            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            
            'lambda_l2': trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 20, 500),

            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),

            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.05, 0.25),

            'feature_fraction': trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                
            'bagging_freq': trial.suggest_int("bagging_freq", 1, 7),
                     
            'verbosity' : 0

        }
    
    start = timer()
    # Perform n_folds cross validation
    if param['boosting_type'] == 'goss':
      param['subsample'] = 1
    else:
      param['subsample'] = trial.suggest_uniform('subsample', 0.5, 1)
    
    cv_results = lgb.cv(param, train_set, num_boost_round = 10000, nfold = 3, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    
    run_time = timer() - start
    
    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])

    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, param, ITERATION_O, n_estimators, run_time])

    
    return loss

# Global variable
global  ITERATION_O

ITERATION_O = 0

# File to save first results
out_file = '/home/jupyter/kshitij/higgs-feature-engg/train/gbm_optuna.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=MAX_EVALS)