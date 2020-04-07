import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV

import csv
from timeit import default_timer as timer

import random

MAX_EVALS = 2
N_FOLDS = 10

# starting with storing the data as data frame
df = pd.read_csv("/home/jupyter/train_test_files/sample.csv")
df.drop( columns='Unnamed: 0', inplace =True)

# making a smaller df for quick testing
df_s, _ = train_test_split(df, random_state = 30, train_size = 0.01)

train_X = df_s.drop(columns = '0')
train_y = df_s['0']

train_set = lgb.Dataset(data=train_X, label = train_y)


def random_objective(params, iteration, n_folds = N_FOLDS):
    """Random search objective function. Takes in hyperparameters
       and returns a list of results to be saved."""

    start = timer()
    # Subsampling (only applicable with 'goss')
    subsample_dist = list(np.linspace(0.5, 1, 100))

    if params['boosting_type'] == 'goss':
        # Cannot subsample with goss
        params['subsample'] = 1.0
    else:
        # Subsample supported for gdbt and dart
        params['subsample'] = random.sample(subsample_dist, 1)[0]    

    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 1000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    end = timer()
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)
    
    # Return list of results
    return [loss, params, iteration, n_estimators, end - start]

# Hyperparameter grid
param_grid = {
                'num_leaves': list(range(16, 196, 4)),
              
                'max_bin': [254],

                'lambda_l1': list(np.linspace(0, 1)),

                'lambda_l2': list(np.linspace(0, 1)),

                'min_data_in_leaf' : list(range(20, 500, 10)),

                'class_weight': [None, 'balanced'],

                'boosting_type': ['gbdt', 'goss', 'dart'],
              
                'learning_rate' : list(np.logspace(np.log(0.05), np.log(0.2), base = np.exp(1), num = 1000)),

                'feature_fraction': list(np.linspace(0.4, 1.0)),

                'bagging_freq': list(range(1,7)),

                'verbosity' : [0]
                }

random.seed(50)

# Dataframe to hold cv results
random_results = pd.DataFrame(columns = ['loss', 'params', 'iteration', 'estimators', 'time'],
                       index = list(range(MAX_EVALS)))

# Iterate through the specified number of evaluations
for i in range(MAX_EVALS):
    
    # Randomly sample parameters for gbm
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
    
    print(params)  
        
    results_list = random_objective(params, i)
    
    # Add results to next row in dataframe
    random_results.loc[i, :] = results_list
    
random_results.to_csv('/home/jupyter/kshitij/higgs-feature-engg/train/random.csv')