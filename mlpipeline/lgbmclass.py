# coding: utf-8
''' This class tunes hyperparamter for LightGBM ML algorithm for Higgs dataset'''

# Draft Codes
# coding: utf-8
''' This class tunes hyperparamter for LightGBM ML algorithm for Higgs dataset'''
import ast 
import csv
import os
from timeit import default_timer as timer
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from hyperopt import STATUS_OK, STATUS_FAIL, hp, tpe, Trials, fmin
import optuna.integration.lightgbm as lgbo
import optuna
import matplotlib.pyplot as plt
import pickle as pkl
# defining constant
MAX_EVALS = 1000
N_FOLDS = 10
NUM_BOOST_ROUNDS = 10000
EARLY_STOPPING_ROUNDS = 100
SEED = 47

RESULT_PATH = 'lgbm.csv'
OBJECTIVE_LOSS = 'binary' # use cross_entropy
EVAL_METRIC = ['auc', 'binary', 'xentropy']

# random search
PARAM_GRID = {
    'num_leaves': list(range(16, 196, 4)),
    'max_bin': [254],
    'lambda_l1': list(np.linspace(0, 1)),
    'lambda_l2': list(np.linspace(0, 1)),
    'min_data_in_leaf' : list(range(20, 500, 10)),
    'boosting_type': ['gbdt', 'goss'],
    #'learning_rate' : list(np.logspace(np.log(0.05), np.log(0.2), base=np.exp(1), num=1000)),
    'feature_fraction': list(np.linspace(0.4, 1.0)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'bagging_freq': list(range(1, 7)),
    'verbosity' : [0],
    'objective' : [OBJECTIVE_LOSS]
    }


# Hyperopt Space
H_SPACE = {
    'num_leaves': hp.quniform('num_leaves', 16, 196, 4),
    'max_bin' : 254, #if using CPU just set this to 254
    'lambda_l1': hp.uniform('lambda_l1', 0.0, 0.5),
    'lambda_l2': hp.uniform("lambda_l2", 0.0, 0.5),
    'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 30, 500, 20),
    'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt',
                                 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                {'boosting_type': 'goss', 'subsample': 1.0}]),
    #'learning_rate' : hp.loguniform('learning_rate', np.log(0.05), np.log(0.25)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'feature_fraction': hp.uniform('feature_fraction', 0.4, 1.0),
    'bagging_freq': hp.uniform('bagging_freq', 1, 7),
    'verbosity' : 0,
    'objective' : OBJECTIVE_LOSS
    }

class Lgbmclass():
    '''Parameter Tuning Class tunes the LightGBM model with different optimization techniques -
    Hyperopt, Optuna and RandomSearch.'''
    iteration = 0
    def __init__(self, x_train, y_train):
        '''Initializes the Parameter tuning class and also initializes LightGBM dataset object
        Parameters
        ----------
        x_train: data (string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse
        or list of numpy arrays) – Data source of Dataset. If string, it represents the path to
        txt file.
        y_train: label (list, numpy 1-D array, pandas Series / one-column DataFrame or None,
        optional (default=None)) – Label of the data.'''

        
        # clear memmory
        gc.collect()
        # File to save first results
        self.out_file = RESULT_PATH
        with open(self.out_file, 'w', newline='') as of_connection:
            writer = csv.writer(of_connection)
            # Write the headers to the file
            writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time','optim_type'])

        self.x_train = x_train
        self.y_train = y_train
        self.train_set = lgb.Dataset(data=x_train, label=y_train)

    def train(self, op_type, device='gpu', diagnostic=False):
        '''
        Trains the object with optimization type
        Parameters
        ----------
        op_type: Type of optimization ('hyperopt', 'optuna' or 'random')
        diagnostic: Bool (Default: False)
        Returns
        -------
        diagnostic = False (default) -> Best parameters from optimization
        diagnostic = True -> Trial list from optimization
        '''
        methodlist = ['hyperopt_space','optuna_space','random_search_space']
        optim_type = op_type + '_space'
        self.device = device
        if optim_type not in methodlist:
            raise TypeError('Otimization type must have a valid space:',
                            '\n\t\t hyperopt, optuna or random_search')
        tuner = getattr(self, optim_type)
        if diagnostic:
            return(tuner())
        else:
            tuner()
            return self.params

    def lgb_crossval(self, params, optim_type):
        '''lgb cross validation model
        Paramters
        ---------
        params: Hyper parameters in dict type from different optimization methods
        optim_type: Is the type of optimization called we use lgb integration for optuna type
        Returns
        ------
        Loss, params, n_estimators, run_time'''
        # initializing the timer
         
        start = timer()
        params['device'] = self.device
        params['is_unbalance'] = True
        if optim_type == 'optuna':
            cv_results = lgbo.cv(params, self.train_set, num_boost_round=NUM_BOOST_ROUNDS,
                                 nfold=N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                 metrics=EVAL_METRIC, seed=SEED)
        else:
            cv_results = lgb.cv(params, self.train_set, num_boost_round=NUM_BOOST_ROUNDS,
                                nfold=N_FOLDS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                metrics=EVAL_METRIC, seed=SEED)
        # store the runtime
        run_time = timer() - start

        # Extract the best score
        best_score = np.max(cv_results['auc-mean'])

        # Loss must be minimized
        loss = 1 - best_score

        # Boosting rounds that returned the highest cv score
        n_estimators = int(np.argmax(cv_results['auc-mean']) + 1) # more explanation
        self.estimator = n_estimators

        # Write to the csv file ('a' means append)
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, self.iteration, n_estimators, run_time, optim_type])

        return loss, params, n_estimators, run_time

    def hyperopt_space(self):
        '''An Lgbm class method to call the hyperopt optimization for the data
        Parameters
        ----------
        fn_name: is the objective function to minimize defined with in the class function
        space: is dictionary type hypeorpt space over which the search is done
        algo: is the type of search algorithm
        trials: Hyperopt base trials object
        Returns
        -------
        result: best parameter that minimizes the fn_name over max_evals = MAX_EVALS FIXED FOR TESTING
        trials: the database in which to store all the point evaluations of the search'''
        print('Running {} rounds of LGBM parameter optimisation using Hyperopt:'.format(MAX_EVALS))
        fn_name, space, algo ='hyperopt_obj', H_SPACE, tpe.suggest
        fn = getattr(self, fn_name)
        try:
            trials = pickle.load(open("lgb_hyperopt","rb"))
        except:
            trials = Trials()
        
        # create checkpoints
        step = 20 # save trial for after every 20 trials
        for i in range(1, MAX_EVALS + 1, step):
            # fmin runs until the trials object has max_evals elements in it, so it can do evaluations in chunks like this
            # each step 'best' will be the best trial so far
            # each step 'trials' will be updated to contain every result
            # you can save it to reload later in case of a crash, or you decide to kill the script
            pickle.dump(trials, open("lgb_hyperopt.pkl", "wb"))
            
            result = fmin(fn=fn, space=space, algo=algo, max_evals=i,
                          trials=trials, rstate=np.random.RandomState(SEED))
        self.params = trials.best_trial['result']['params']
        self.params['n_estimators'] = self.estimator
        return result, trials

    def hyperopt_obj(self, params):
        """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

        optim_type = 'Hyperopt'
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
        loss, params, n_estimators, run_time = self.lgb_crossval(params, optim_type)

        # Dictionary with information for evaluation
        return {'loss':loss, 'params':params, 'iteration':self.iteration,
                'estimators':n_estimators, 'train_time':run_time, 'status':STATUS_OK}
                
    def optuna_space(self):
        '''An Optuna class method to call the Optuna optimization for the data
        Parameters
        ----------
        fn_name: is the optuna objective function to minimize defined with in the class function
        direction: to indicate if the objective function is a loss to be minimized or gain to be maximized
        n_trials: Optuna evaluation roundd
        Returns
        -------
        study: Optuna study object
        '''
        
        print('Running {} rounds of LGBM parameter optimisation using Optuna:'.format(MAX_EVALS))
        fn_name = 'optuna_obj'
        fn = getattr(self, fn_name)
        try:
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(fn, n_trials=MAX_EVALS)
        except Exception as e:
            return {'exception': str(e)}
        self.params = study.best_params
        self.params['n_estimators'] = self.estimator
        return study

    def optuna_obj(self, trial):
        '''Defining the parameters space inside the function for optuna optimization'''
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 16, 196, 4),
            'max_bin' : trial.suggest_int('max_bin', 254, 255, 1),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 20, 500),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
            # removed 'dart'
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.05, 0.25),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin',20000, 300000, 20000),
            'feature_fraction': trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            'bagging_freq': trial.suggest_int("bagging_freq", 1, 7),
            'verbosity' : 0,
            'objective' : OBJECTIVE_LOSS
                }

        optim_type = 'Optuna'
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

        loss, params, _, _ = self.lgb_crossval(params, optim_type)

        return loss

    def random_search_space(self):
        '''Random search space'''
        print('Running {} rounds of LGBM parameter optimisation using Random Search:'.format(MAX_EVALS))
        # Dataframe to hold cv results
        space = PARAM_GRID
        random_results = pd.DataFrame(columns=['loss', 'params', 'iteration', 'estimators',
                                               'time'], index=list(range(MAX_EVALS)))

        # Iterate through the specified number of evaluations
        for i in range(MAX_EVALS):

            # Randomly sample parameters for gbm
            params = {key: random.sample(value, 1)[0] for key, value in space.items()}
            results_list = self.randomsrch_obj(params, i)

            # Add results to next row in dataframe
            random_results.loc[i, :] = results_list
        #sort values by the loss
        random_results.sort_values('loss', ascending = True, inplace = True)
        self.params = random_results.loc[0, 'params']
        self.params['n_estimators'] = self.estimator
        return random_results

    def randomsrch_obj(self, params, iteration):
        """Random search objective function. Takes in hyperparameters and returns a list
        of results to be saved."""
        optim_type = 'Random'
        self.iteration += 1
        random.seed(SEED)
        # Subsampling (only applicable with 'goss')
        subsample_dist = list(np.linspace(0.5, 1, 100))

        if params['boosting_type'] == 'goss':
            # Cannot subsample with goss
            params['subsample'] = 1.0
        else:
            # Subsample supported for gdbt and dart
            params['subsample'] = random.sample(subsample_dist, 1)[0]

        # Perform n_folds cross validation
        loss, params, n_estimators, run_time = self.lgb_crossval(params, optim_type)

        # Return list of results
        return [loss, params, iteration, n_estimators, run_time]

    def test(self, x_test, y_test):
        """This function trains the model on best paramters and estimators
        
        Parameters
        ----------
        x_test: test set; y_test: test label
        Return
        ----------
        Predict Proba for test data"""

        self.test_x, self.test_y = x_test, y_test
        param_df = pd.read_csv(RESULT_PATH)
        param_df.sort_values('loss', ascending = True, inplace = True)

        best = ast.literal_eval(param_df.loc[0, 'params'])
        best['n_estimators'] = int(param_df.loc[0, 'estimators'])
        optim_type = param_df.loc[0, 'optim_type']
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_data_in_leaf',
                               'max_bin', 'bagging_freq']:
            best[parameter_name] = int(best[parameter_name])
        self.gbm = lgb.train(best, self.train_set,
                             feature_name=['f' + str(i + 1) for i in range(self.x_train.shape[-1])])
        self.pred = self.gbm.predict(x_test)
        print("Model will be trained with best parameters obtained from {} ... \n\n\n".format(optim_type))
        print("Model trained on the following parameters: \n{}".format(best))
        print('Plotting feature importances...')
        ax = lgb.plot_importance(self.gbm, max_num_features=10)
        plt.savefig(os.path.join("figs",'lgb_'+optim_type+'feature_importance.png'))
        return self.pred