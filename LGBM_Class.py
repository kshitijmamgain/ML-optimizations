import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV

import csv
from timeit import default_timer as timer
from hyperopt import STATUS_OK, STATUS_FAIL, hp, tpe, Trials, fmin
from hyperopt.pyll.stochastic import sample
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

# drop last columns

def col_keep(df):
  return df.drop(columns = list(map(str,range(22,29))), inplace = True) # removing 7 last columns

MAX_EVALS = 1000
N_FOLDS = 3
NUM_BOOST_ROUNDS = 10000
EARLY_STOPPING_ROUNDS = 100
SEED = 47


class Parameter_Tuning():
  '''
  Parameter Tuning Class tunes the LightGBM model with different optimization techniques -
  Hyperopt, Optuna and RandomSearch.
  At present method for CV using Hyperopt is defined
  '''  
  def __init__(self, x_train, y_train):
    '''
    Initializes the Parameter tuning class and also initializes LightGBM dataset object
    
    Parameters
    ----------
    x_train: data (string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse or list of numpy arrays) – Data source
    of Dataset. If string, it represents the path to txt file.
    
    y_train: label (list, numpy 1-D array, pandas Series / one-column DataFrame or None, optional (default=None)) – Label of the 
    data.
    '''
    self.x_train = x_train
    self.y_train = y_train
    self.train_set = lgb.Dataset(data=train_X, label = train_y)
  
  def hyperopt_space(self,fn_name, space, algo, trials):
     '''
     A method to call the hyperopt for the data
     
     Parameters
     ----------
     fn_name: is the objective function to minimize defined with in the class function
     space: is dictionary type hypeorpt space over which the search is done
     algo: is the type of search algorithm
     trials: Hyperopt base trials object
     
     Returns
     -------
     result: best parameter that minimizes the fn_name over max_evals = 50 FIXED FOR TESTING
     trials: the database in which to store all the point evaluations of the search
     '''
     
     fn = getattr(self, fn_name)
     try:
       result = fmin(fn = fn, space = space, algo = algo, max_evals = 50, 
                   trials = trials, rstate = np.random.RandomState(50))
     except Exception as e:
        return {'status': STATUS_FAIL, 'exception': str(e)}
     return result, trials
  
  def lgbm_cv(self,params, n_folds = 3):
    """
    Objective function for Gradient Boosting Machine Hyperparameter Optimization
    
    Parameters
    ----------
    params: takes the sampled space from hyperopt_space function
    n_folds: number of folds for cross-validation, currently FIXED AT 3
    
    Returns
    -------
    loss: retunrs the minimu value for 1 - auc-mean for cross validation
    """
        
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample


    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_data_in_leaf', 'max_bin', 'bagging_freq']:
      params[parameter_name] = int(params[parameter_name])
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, self.train_set, num_boost_round = 10000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = ['auc','binary','xentropy'], seed = 50)
    
    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
       
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# Hyperopt Space
hyperopt_space = {              
            'num_leaves': hp.quniform('num_leaves', 16, 196, 4),

            'max_bin' : hp.quniform('max_bin', 253, 254, 1), #if using CPU just set this to 254

            'lambda_l1': hp.uniform('lambda_l1', 0.0, 1.0),
            
            'lambda_l2': hp.uniform("lambda_l2", 0.0, 1.0),
            
            'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 20, 500, 10),

            #'class_weight': hp.choice('class_weight', [None, 'balanced']), #class weigth is valid only for RF?
            
            'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                         #{'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)}, taking long to tune?
                                                         {'boosting_type': 'goss', 'subsample': 1.0}]),
                 
            'learning_rate' : hp.loguniform('learning_rate', np.log(0.05), np.log(0.25)),

            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),

            'feature_fraction': hp.uniform('feature_fraction', 0.4, 1.0),
                     
            'bagging_freq': hp.uniform('bagging_freq', 1, 7), 
                     
            'verbosity' : 0,
            
            }

obj = Parameter_Tuning(train_X,train_y)

lgb_ho = obj.hyperopt_space(fn_name = 'lgbm_cv', space = hyperopt_space,algo= tpe.suggest, trials = Trials())