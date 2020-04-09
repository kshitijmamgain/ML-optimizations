##Some requirements below are redundant!!!
import pandas as pd
import sklearn
import numpy as np
from matplotlib import pyplot as plt
import pandas.util.testing as tm
from numpy.random import uniform
import catboost as cb
import gc
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.stochastic import sample
import warnings
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score,auc, roc_auc_score, roc_curve,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import csv
from timeit import default_timer as timer
import random
from catboost import CatBoostClassifier
from catboost import *
from numpy.random import RandomState

##Data
__FILE_LOC = "c:/users/hajyhass/downloads/train_test_files_sample.csv"
df_train = pd.read_csv(__FILE_LOC,header=0)
df_reduced = df_train.drop(columns= ["0"])
data_selected= df_reduced[:][:]
labels_selected = df_train["0"][:]
indices = uniform(0,df_reduced.shape[0], 5000)
indices_list = [int(x) for x in indices]
sampled_data = df_reduced.iloc[indices_list,]
sampled_data=sampled_data.drop(columns='Unnamed: 0')
sampled_target = df_train["0"][indices_list] 
warnings.filterwarnings('ignore')
# making a smaller df for quick testing
df_s, _ = train_test_split(df_train, random_state = 30, train_size = 0.1)
train_X = sampled_data
train_y = sampled_target



#Configure Defaults
MAX_EVALS = 5
N_FOLDS = 3
NUM_BOOST_ROUNDS = 100
EARLY_STOPPING_ROUNDS = 100
SEED = 47
#GLOBAL HYPEROPT PARAMETERS ->These must be in Configure file already
NUM_EVALS = 10 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round
#CATBOOST PARAMETERS
CB_MAX_DEPTH = 8 #maximum tree depth in CatBoost
#OBJECTIVE_CB_REG = 'MAE' #CatBoost regression metric
#OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric


##The Object Orineted below is modified on Kshitij's original script
class Parameter_Tuning:
    
    '''The Parameter_Tuning Class, optimizes via Hyperopt.The class is composed of different functions including:
    __init__() that initializes the input arguments,
    hyperopt_space() which runs a global optimization over a pre-defined parameter speace,
    ctb_cv() which is the main catboost technology'''
    '''The Parameter_Tuning Class, needs other functions including:
    tuner() to feed optimal values into,
    predict() to predict the results based on most optimized parameters,
    the metric() to evaluate the results
    same pattern must be repeated for Optuna'''

    def __init__(self, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train
        self.train_set = cb.Pool(self.x_train, self.y_train)

    def hyperopt_space(self, fn_name, space, algo, trials):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=MAX_EVALS,
                   trials=trials, rstate=np.random.RandomState(50))
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}
        return result, trials

    def ctb_cv(self, params, n_folds=3):
        
        """ What are we achieving below?
        Sometimes we have a dictionary inside which there are different types of input, for example one choice
        is a 1-key dict and the other choices are 2-key dicts. Therefore, we msut define a conditional statement
        to identify which type of dictionary we are dealing with, and if it is of inconsistent type, we normalize it"""

        # Extract the bootstrap_type

        if params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
            print(params['bootstrap_type'])
            params['bagging_temperature'] = params['bootstrap_type']['bagging_temperature']
            print(params['bagging_temperature'])
            params['bootstrap_type'] = params['bootstrap_type']['bootstrap_type']
            print(params['bootstrap_type'])
        else:
            params['bootstrap_type'] = params['bootstrap_type']['bootstrap_type']
            print(params['bootstrap_type'])

        if params['grow_policy']['grow_policy'] == 'Lossguide':
            params['max_leaves'] = params['grow_policy']['max_leaves']
            print(params['max_leaves'])
            params['grow_policy'] = params['grow_policy']['grow_policy']
            score_function = 'NewtonL2'    
            ###when 'grow_policy' is set to 'lossguide', the only possible score_function is 'NewtonL2'
        else:
            params['grow_policy'] = params['grow_policy']['grow_policy']
            print(params['grow_policy'])

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['l2_leaf_reg', 'depth', 'border_count']:
            params[parameter_name] = int(params[parameter_name])
        
        cv_results = cb.cv(self.train_set, params, fold_count=N_FOLDS, num_boost_round=NUM_BOOST_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, stratified=True, partition_random_seed=42,
                plot=True)
        
        # Extract the best score
        best_score = np.max(cv_results['test-RMSE-mean'])

        # Loss must be minimized
        loss = 1 - best_score

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}


hyperopt_space = {
            'l2_leaf_reg': hp.qloguniform('l2_leaf_reg', 0, 2, 1),
            'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
            'depth': hp.quniform('depth', 1, CB_MAX_DEPTH, 1),
            'border_count': hp.quniform('border_count', 32, 255, 1),
            'bootstrap_type': hp.choice('bootstrap_type', [{'bootstrap_type': 'Bayesian', 'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
                                {'bootstrap_type': 'Bernoulli'}]),
            'grow_policy': hp.choice('grow_policy', [{'grow_policy': 'SymmetricTree'}, {'grow_policy': 'Depthwise'},
                                {'grow_policy': 'Lossguide', 'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]),
            'score_function': hp.choice('score_function', ['Cosine']),
            'eval_metric': hp.choice('eval_metric', ['MAE', 'RMSE', 'Poisson']),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 50, 1),
            'random_strength': hp.loguniform('random_strength', np.log(0.005), np.log(5)),
            # 'score_function' = ['Correlation', 'L2', 'NewtonCorrelation', 'NewtonL2','LOOL2','Cosine','SatL2','SolarL2']
            # NewtonL2 is the only possible choice for Lossguide
            # The other choices explicitely belong to GPU, only 'Cosine' applies to CPU
            'rsm': hp.uniform('rsm', 0.1, 1),
            'od_type': hp.choice('od_type', ['IncToDec', 'Iter']),
            'task_type': 'CPU',
            # 'thread_count':-1
            'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', ['No', 'AnyImprovement'])
            #AnyImprovement — Reduces the descent step up to the point when the loss function value is smaller than it was on the previous step.
            }

obj = Parameter_Tuning(train_X, train_y)

ctb_ho = obj.hyperopt_space(fn_name='ctb_cv', space=hyperopt_space, algo=tpe.suggest, trials=Trials())