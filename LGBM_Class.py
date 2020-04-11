
# coding: utf-8
''' This class tunes hyperparamter for LightGBM ML algorithm for Higgs dataset'''
import csv
from timeit import default_timer as timer
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
#from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from hyperopt import STATUS_OK, STATUS_FAIL, hp, tpe, Trials, fmin
#from hyperopt.pyll.stochastic import sample
#import optuna.integration.lightgbm as lgbo
import optuna

# starting with storing the data as data frame
df = pd.read_csv("/home/jupyter/train_test_files/sample.csv")
df.drop(columns='Unnamed: 0', inplace=True)

# making a smaller df for quick testing
df_s, _ = train_test_split(df, random_state=30, train_size=0.01)
train_X = df_s.drop(columns='0')
train_y = df_s['0']

# drop last columns
def col_keep(df):
    '''Removes the last 7 columns'''
    return df.drop(columns=list(map(str, range(22, 29))), inplace=True) # removing 7 last columns


MAX_EVALS = 5
N_FOLDS = 3
NUM_BOOST_ROUNDS = 10000
EARLY_STOPPING_ROUNDS = 100
SEED = 47

# random search
PARAM_GRID = {
    'num_leaves': list(range(16, 196, 4)),
    'max_bin': [254],
    'lambda_l1': list(np.linspace(0, 1)),
    'lambda_l2': list(np.linspace(0, 1)),
    'min_data_in_leaf' : list(range(20, 500, 10)),
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss'],
    'learning_rate' : list(np.logspace(np.log(0.05), np.log(0.2), base=np.exp(1), num=1000)),
    'feature_fraction': list(np.linspace(0.4, 1.0)),
    'bagging_freq': list(range(1, 7)),
    'verbosity' : [0]
    }


# Hyperopt Space
H_SPACE = {
    'num_leaves': hp.quniform('num_leaves', 16, 196, 4),
    'max_bin' : hp.quniform('max_bin', 253, 254, 1), #if using CPU just set this to 254
    'lambda_l1': hp.uniform('lambda_l1', 0.0, 1.0),
    'lambda_l2': hp.uniform("lambda_l2", 0.0, 1.0),
    'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 20, 500, 10),
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt',
                                 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                #{'boosting_type': 'dart',
                                #'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                {'boosting_type': 'goss', 'subsample': 1.0}]),
    'learning_rate' : hp.loguniform('learning_rate', np.log(0.05), np.log(0.25)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'feature_fraction': hp.uniform('feature_fraction', 0.4, 1.0),
    'bagging_freq': hp.uniform('bagging_freq', 1, 7),
    'verbosity' : 0
    }

class Parameter_Tuning():
    '''Parameter Tuning Class tunes the LightGBM model with different optimization techniques -
    Hyperopt, Optuna and RandomSearch. At present method for CV using Hyperopt is defined'''
    def __init__(self, x_train, y_train):
        '''Initializes the Parameter tuning class and also initializes LightGBM dataset object
        Parameters
        ----------
        x_train: data (string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse
        or list of numpy arrays) – Data source of Dataset. If string, it represents the path to
        txt file.
        y_train: label (list, numpy 1-D array, pandas Series / one-column DataFrame or None,
        optional (default=None)) – Label of the data.'''

        self.iteration = 0

        # File to save first results
        self.out_file = 'gbm_trials.csv'
        of_connection = open(self.out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
        of_connection.close()

        self.x_train = x_train
        self.y_train = y_train
        self.train_set = lgb.Dataset(data=train_X, label=train_y)

    def lgb_crossval(self, params):
        '''lgb cross validation model'''
        # initializing the timer
        start = timer()

        cv_results = lgb.cv(params, self.train_set, num_boost_round=NUM_BOOST_ROUNDS, nfold=N_FOLDS,
                            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                            metrics=['auc', 'binary', 'xentropy'], seed=SEED)
        # store the runtime
        run_time = timer() - start

        # Extract the best score
        best_score = np.max(cv_results['auc-mean'])

        # Loss must be minimized
        loss = 1 - best_score

        # Boosting rounds that returned the highest cv score
        n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

        # Write to the csv file ('a' means append)
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, self.iteration, n_estimators, run_time])

        return loss, params, n_estimators, run_time

    def hyperopt_space(self, fn_name, space, algo=tpe.suggest, trials=Trials()):
        '''A method to call the hyperopt optimization for the data
        Parameters
        ----------
        fn_name: is the objective function to minimize defined with in the class function
        space: is dictionary type hypeorpt space over which the search is done
        algo: is the type of search algorithm
        trials: Hyperopt base trials object
        Returns
        -------
        result: best parameter that minimizes the fn_name over max_evals = 50 FIXED FOR TESTING
        trials: the database in which to store all the point evaluations of the search'''
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=MAX_EVALS,
                          trials=trials, rstate=np.random.RandomState(50))
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}
        return result, trials

    def optuna_space(self, fn_name):
        '''Optuna search space'''
        fn = getattr(self, fn_name)
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(fn, n_trials=5)
        except Exception as e:
            return {'exception': str(e)}
        return study

    def random_space(self, space):
        '''Random search space'''
        # Dataframe to hold cv results
        random_results = pd.DataFrame(columns=['loss', 'params', 'iteration', 'estimators',
                                               'time'], index=list(range(MAX_EVALS)))

        # Iterate through the specified number of evaluations
        for i in range(MAX_EVALS):

            # Randomly sample parameters for gbm
            params = {key: random.sample(value, 1)[0] for key, value in space.items()}
            results_list = self.randomsrch_obj(params, i)

            # Add results to next row in dataframe
            random_results.loc[i, :] = results_list
        return random_results

    def hyperopt_obj(self, params):
        """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

        self.iteration += 1

        # Retrieve the subsample if present otherwise set to 1.0
        subsample = params['boosting_type'].get('subsample', 1.0)

        # Extract the boosting type
        params['boosting_type'] = params['boosting_type']['boosting_type']
        params['subsample'] = subsample

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_data_in_leaf',
                               'max_bin', 'bagging_freq']:
            params[parameter_name] = int(params[parameter_name])

        # Perform n_folds cross validation
        loss, params, n_estimators, run_time = self.lgb_crossval(params)

        # Dictionary with information for evaluation
        return {'loss':loss, 'params':params, 'iteration':self.iteration,
                'estimators':n_estimators, 'train_time':run_time, 'status':STATUS_OK}

    def optuna_obj(self, trial):
        '''Defining the parameters space inside the function for optuna optimization'''
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 16, 196, 4),
            'max_bin' : trial.suggest_discrete_uniform('max_bin', 63, 255, 64),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 20, 500),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
            # removed 'dart'
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.05, 0.25),
            'feature_fraction': trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            'bagging_freq': trial.suggest_int("bagging_freq", 1, 7),
            'verbosity' : 0
                }

        self.iteration += 1

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'min_data_in_leaf',
                               'max_bin', 'bagging_freq']:
            params[parameter_name] = int(params[parameter_name])

        # Perform n_folds cross validation
        if params['boosting_type'] == 'goss':
            params['subsample'] = 1
        else:
            params['subsample'] = trial.suggest_uniform('subsample', 0.5, 1)

        loss, params, _, _ = self.lgb_crossval(params)

        return loss

    def randomsrch_obj(self, params, iteration):
        """Random search objective function. Takes in hyperparameters and returns a list
        of results to be saved."""
        self.iteration += 1

        # Subsampling (only applicable with 'goss')
        subsample_dist = list(np.linspace(0.5, 1, 100))

        if params['boosting_type'] == 'goss':
            # Cannot subsample with goss
            params['subsample'] = 1.0
        else:
            # Subsample supported for gdbt and dart
            params['subsample'] = random.sample(subsample_dist, 1)[0]

        # Perform n_folds cross validation
        loss, params, n_estimators, run_time = self.lgb_crossval(params)

        # Return list of results
        return [loss, params, iteration, n_estimators, run_time]

obj = Parameter_Tuning(train_X, train_y)

lgb_ho = obj.hyperopt_space(fn_name='lgbm_cv', space=H_SPACE,
                            algo=tpe.suggest, trials=Trials())
    