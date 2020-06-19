import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample
from sklearn.metrics import log_loss
from timeit import default_timer as timer
import pickle
import random
import seaborn as sns
import shap
import csv
import os
from sklearn.metrics import f1_score

def f1_eval(pred, dmatrix):
    y = dmatrix.get_label()
    f1 = f1_score(y, np.round(pred))
    return 'F1', 1-f1

class XGBoostModel():

    """
    Executes methods that split a dataset into training and testing sets,
    runs an XGBoost classifier with cross-validation, and attempts to
    optimize the hyperparameters using three different methods: HyperOpt,
    Optuna and RandomSearch
    """

    def __init__(self, x_train, y_train, max_evals, n_fold, num_boost_rounds,
                 early_stopping_rounds, seed, GPU):

      """
      Initializes an instance of the XGB_Higgs Class

      Parameters
      ----------
      x_train: numpy array, Pandas dataframe or Series
        Contains observations for predictor features to be used in training
      y_train: numpy array, Pandas dataframe or Series
        Contains with observations for target feature to be used in training
      max_evals: int
        The number of trials to run the optimization algorithms
      n_fold: int
        Number of cross-validation folds to use during validation
      num_boost_rounds: int
        Number of boosting trees/estimators to use when training
      early_stopping_rounds: int
        Threshold for num_boost_rounds. Training will be stopped if the evaluation metric
        does not improve for this many rounds
      seed: int
        Sets the random seed for all algorithms for reproducibility
      GPU: bool
        Enable GPU usage and include GPU specific-parameters to parameter search space

      """
      if not isinstance(max_evals, int):
        raise TypeError('Number of evaluations must be an integer')
      if not isinstance(n_fold, int):
        raise TypeError('Number of cross-validation folds must be an integer')
      if not isinstance(num_boost_rounds, int):
        raise TypeError('Number of boosting rounds must be an integer')
      if not isinstance(early_stopping_rounds, int):
        raise TypeError('Number of early stopping rounds must be an integer')
      if not isinstance(seed, int):
        raise TypeError('Seed must be an integer')
      if not isinstance(GPU, bool):
        raise TypeError('Value of GPU must be set to either True or False')

      self.max_evals = max_evals
      self.nfold = n_fold
      self.num_boost_rounds = num_boost_rounds
      self.early_stopping_rounds = early_stopping_rounds
      self.X_train, self.y_train = x_train, y_train
      self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
      label = self.dtrain.get_label()
      self.ratio = float(np.sum(label == 0)) / np.sum(label == 1)
      self.dtest = None
      self.best_params = None
      self.best_model = None
      self.output = None
      self.seed = seed
      self.GPU = GPU
      self.trained = False
      self.tested = False
      
    def cross_validation(self, space):

      """
      Applies cross_validation for the XGBoost algorithm

      Parameters
      ----------
      space: dict
        A dictionary containing the parameters to use when training the model

      Returns
      -------
      results: dict
        A dictionary containing the following attributes:
          'loss': the mean value of the loss function across n-fold
          cross-validations
          'variance':  the variance of the loss function across n-fold
          cross-validation
          'params': the parameters used to train the model
          'n_estimators': the number of boosting rounds used
          'time': the time it took to run the model
      """
    
      if not isinstance(space, dict):
        raise TypeError('Parameters must be provided as a dictionary')

      start = timer()
      cv_results = xgb.cv(space,
                          self.dtrain,
                          num_boost_round=self.num_boost_rounds,
                          early_stopping_rounds=self.early_stopping_rounds,
                          stratified=True,
                          feval=f1_eval,
                          verbose_eval=True,
                          nfold=self.nfold,
                          metrics=space['eval_metric']
#                           seed=self.seed
                         )
      end = timer()
      cv_score = np.min(cv_results['test-F1-mean'])
      cv_var = np.min(cv_results['test-F1-std'])**2
      n_estimators = int(np.argmin(cv_results['test-F1-mean']) + 1)
      results = {'loss': cv_score, 'variance': cv_var, 'params': space,
                 'n_estimators': n_estimators, 'time': end - start}

      return results

    def hyperopt_tuning(self):

      """
      Applies the HyperOpt algorithm to tune the hyperparameters of the XGBoost
      model and stores the results in a dataframe

      Returns
      -------
      best: dict
        A dictionary containing the set of parameters that resulted in the best
        value of the loss function using the HyperOpt optimization algorithm.
      """
      print('Starting HyperOpt hyperparameter tuning...')
      self.trials = pd.DataFrame(columns=['params', 'loss',
                                                   'variance', 'n_estimators',
                                                   'time'])

      def hyperopt_params():

        """
        Defines the hyperparameter search space for the HyperOpt algorithm

        Returns
        ----------
        params: dict
          A dictionary containing the search-space for the Hyperopt
          optimization algorithm
        """


        if self.GPU:
          tree_method = [{'tree_method': 'gpu_hist',
                          'single_precision_histogram': hp.choice(
                                  'single_precision_histogram', [True, False]),
                          'deterministic_histogram': hp.choice(
                                  'deterministic_histogram', [True, False])
                         }]
          subsample = hp.quniform('subsample', 0.1, 1, 0.1)

        else:
          tree_method = [
                        #  {'tree_method': 'hist',
                        #   'grow_policy': hp.choice('grow_policy', ['lossguide', 'depthwise'])
                        #  },
                         {'tree_method': 'approx'}]
          subsample = hp.quniform('subsample', 0.5, 1, 0.1)

        params = {
                  'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'verbosity': 1,
                  'disable_default_eval_metric': 1,
                  'booster': 'gbtree',
                  'reg_lambda': hp.loguniform('reg_lambda', -3*np.log(10), 2*np.log(10)),
                  'reg_alpha': hp.loguniform('reg_alpha', -3*np.log(10), 2*np.log(10)),
                  'max_depth': hp.choice('max_depth', np.arange(4, 12,
                                                                dtype=int)),
                  'gamma': hp.loguniform('gamma', -3*np.log(10), 2*np.log(10)),
                  'subsample': subsample,
                  'sampling_method': 'uniform',
                  'colsample_bytree': hp.quniform('colsample_bytree',
                                                  0.5, 1, 0.1),
                  'colsample_bylevel': hp.quniform('colsample_bylevel',
                                                   0.5, 1, 0.1),
                  'colsample_bynode': hp.quniform('colsample_bynode',
                                                  0.5, 1, 0.1),
                  'nthread': 8,
                  'tree_method': hp.choice('tree_method', tree_method),
                  'scale_pos_weight': self.ratio,
                  'predictor': 'cpu_predictor'
                  }

        return params

      space = hyperopt_params()

      def hyperopt_objective(space):

        """
        Defines the objective function to be optimized by the HyperOpt
        algorithm

        Parameters
        ----------
        space: dict
          A dictionary of parameters to use for each trial of the HyperOpt
          optimization algorithm

        Returns
        -------
        result_dict: dict
          A dictionary containing the loss for the current set of parameters,
          the status of the current trial and a dictionary of parameters used
          for the trial
        """
        if not isinstance(space, dict):
          raise TypeError('Parameters must be provided as a dictionary')
        
        space['max_depth'] = int(space['max_depth'])

        # if space['tree_method']['tree_method'] == 'hist':
        #       grow_policy = space['tree_method'].get('grow_policy')
        #       space['grow_policy'] = grow_policy
        #       if space['grow_policy']=='lossguide':
        #         space['max_leaves'] = 2**space['max_depth']-1

        if space['tree_method']['tree_method'] == 'gpu_hist':
              single_precision_histogram = space['tree_method'].get(
                                                  'single_precision_histogram')
              deterministic_histogram = space['tree_method'].get(
                                                    'deterministic_histogram')
              space['single_precision_histogram'] = single_precision_histogram
              space['deterministic_histogram'] = deterministic_histogram
              space['sampling_method'] = 'gradient_based'
              space['predictor'] = 'gpu_predictor'

        space['tree_method'] = space['tree_method']['tree_method']

        print('Training with params: ')
        print(space)

        results = self.cross_validation(space)
        self.trials = self.trials.append(results, ignore_index=True)

        current_time = (timer()-self.time)/3600
        print('\n')
        print('Time elapsed so far: '+ str(current_time) + ' hours')
        print('\n')

        result_dict = {'loss': results['loss'],
                       'status': STATUS_OK,
                       'parameters': results['params']}

        return (result_dict)

      trials = Trials()
      self.time = timer()

      try:
        optimize = fmin(fn=hyperopt_objective,
                      space=space,
                      algo=tpe.suggest,
                      trials=trials,
                      max_evals=self.max_evals
#                       rstate=np.random.RandomState(seed=self.seed)
                       )
      except:
        pass
      self.opt_time = (timer() - self.time)/3600
        
      if not os.path.exists('XGBoost_trials'):
        os.makedirs('XGBoost_trials')
      self.trials.to_csv('XGBoost_trials/hyperopt_trials.csv')

      best = self.trials[['params', 'loss', 'n_estimators']].sort_values(
                                    by='loss', ascending=True).reset_index().loc[0].to_dict()
        
      self.estimators = best['n_estimators']  
      best = best['params']
      self.best_params = best
      print('The best HyperOpt hyperparameters are: ')
      print(best)
      print('\n')
      return best

    def optuna_tuning(self):

      """
      Applies the Optuna algorithm to tune the hyperparameters of the XGBoost
      model and stores the results in a dataframe

      Returns
      ----------
      best: dict
        A dictionary containing the set of parameters that resulted in the best
        value of the loss function using the Optuna optimization algorithm.
      """
      optuna.logging.enable_propagation()
      optuna.logging.disable_default_handler()
      print('Starting Optuna hyperparameter tuning...')
      self.trials = pd.DataFrame(columns=['params', 'loss', 'variance',
                                                 'n_estimators', 'time'])

      def optuna_objective(trial):

        """
        Defines the objective function to be optimized by the Optuna algorithm

        Parameters
        ----------
        trial: Optuna trial object
          An Optuna trial that iterates over a set of hyperparameters to
          minimize the objective function

        Returns
        -------
        cv_score: float
          The value of the loss function cross-validated over n-folds for the
          current set of hyperparameters for the trial
        """
        if self.GPU:
          tree_method_list = ['gpu_hist']
          sampling_method = ['gradient_based']
          predictor = ['gpu_predictor']
          subsample = trial.suggest_uniform('subsample', 0.1, 1)
        
        else:
          tree_method_list = ['approx'
          # , 'hist'
          ]
          sampling_method = ['uniform']
          predictor = ['cpu_predictor']
          subsample = trial.suggest_uniform('subsample', 0.5, 1)

        space = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 1,
            'disable_default_eval_metric': 1,
            'booster': 'gbtree',
            #'booster': trial.suggest_categorical('booster',['gbtree']),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1e2),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1e2),
            #'max_delta_step': trial.suggest_int('max_delta_step', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            #'eta': trial.suggest_uniform('eta', 0.025, 0.5),
            'gamma': trial.suggest_loguniform('gamma', 1e-3, 1e2),
            'subsample': subsample,
            #'grow_policy': trial.suggest_categorical('grow_policy',
             #                                      ['depthwise', 'lossguide']),
            #'min_child_weight': trial.suggest_uniform('min_child_weight',
             #                                         1, 10),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree',
                                                      0.5, 1),
            'colsample_bylevel': trial.suggest_uniform('colsample_bylevel',
                                                       0.5, 1),
            'colsample_bynode': trial.suggest_uniform('colsample_bynode',
                                                      0.5, 1),
            'nthread': 8,
            'tree_method': trial.suggest_categorical('tree_method',
                                                     tree_method_list),
            'scale_pos_weight': self.ratio,
            'sampling_method': trial.suggest_categorical('sampling_method',
                                                     sampling_method),
            'predictor': trial.suggest_categorical('predictor',
                                                     predictor)
                  }

        # if space['grow_policy'] == 'lossguide':
        #     space['max_leaves'] = trial.suggest_int('max_leaves', 0, 10)

        # if space['booster'] == 'dart':
        #     space['sample_type'] = trial.suggest_categorical(
        #                                 'sample_type', ['uniform', 'weighted'])
        #     space['normalize_type'] = trial.suggest_categorical(
        #                                   'normalize_type', ['tree', 'forest'])
        #     space['rate_drop'] = trial.suggest_uniform('rate_drop', 0, 1)
        #     space['skip_drop'] = trial.suggest_uniform('skip_drop', 0, 1)

        # if space['tree_method'] == 'hist':
        #     space['max_bin'] = trial.suggest_categorical('max_bin',
        #                                              [2**7, 2**8, 2**9, 2**10])

        # if space['tree_method'] == 'approx':
        #     space['sketch_eps'] = trial.suggest_uniform('sketch_eps', 0.01, 0.99)

        if space['tree_method'] == 'gpu_hist':
            space['single_precision_histogram'] = trial.suggest_categorical(
                                   'single_precision_histogram', [True, False])
            space['deterministic_histogram'] = trial.suggest_categorical(
                                    'deterministic_histogram', [True, False])
            space['max_bin'] = trial.suggest_categorical('max_bin',
                                                     [2**7, 2**8, 2**9, 2**10])

        print('Training with params: ')
        print(space)

        results = self.cross_validation(space)
        self.trials = self.trials.append(results, ignore_index=True)

        current_time = (timer()-self.time)/3600
        print('\n')
        print('Time elapsed so far: '+ str(current_time) + ' hours')
        print('\n')

        return results['loss']

      self.time = timer()

      try:
        study = optuna.create_study(direction='minimize'
#                                   sampler=optuna.samplers.TPESampler(
#                                       seed=self.seed)
                                 )
        optimize = study.optimize(optuna_objective, n_trials=self.max_evals)
      except:
        pass
      self.opt_time = (timer() - self.time)/3600

      if not os.path.exists('XGBoost_trials'):
        os.makedirs('XGBoost_trials')
      self.trials.to_csv('XGBoost_trials/optuna_trials.csv')

      best = self.trials[['params', 'loss']].sort_values(
                                    by='loss', ascending=True).reset_index().loc[0].to_dict()
      best = best['params']
      self.best_params = best

      print('The best Optuna hyperparameters are: ')
      print(best)
      print('\n')
      return best

    def random_search_tuning(self):

      """
      Applies the Random Search algorithm to tune the hyperparameters of the
      XGBoost model and stores the results in a dataframe

      Returns
      -------
      best: dict
        A dictionary containing the set of parameters that resulted in the best
        value of the loss function using the Random Search optimization
        algorithm.
      """
#       random.seed(self.seed)
      print('Starting Random Search hyperparameter tuning...')
      self.trials = pd.DataFrame(columns=['params',
                                          'loss',
                                          'variance',
                                          'n_estimators',
                                          'time'])

      def random_params():

        """
        Defines the hyperparameter search space for the Random Search algorithm

        Returns
        -------
        params: dict
          A dictionary containing the search-space for the Hyperopt
          optimization algorithm
        """

        if self.GPU:
          tree_method_list = ['gpu_hist']
        else:
          tree_method_list = ['approx'
                              #, 'hist'
                              ]

        params = {
            'objective': ['binary:logistic'],
            'eval_metric': ['logloss'],
            'disable_default_eval_metric': [1],
            'booster': ['gbtree'],
            'reg_lambda': np.arange(1e-3, 1e2),
            'reg_alpha': np.arange(1e-3, 1e2),
            'verbosity': [1],
            #'max_delta_step': np.arange(1, 10, 1),
            'max_depth': np.arange(4, 12),
            #'eta': np.arange(0.025, 0.5, 0.025),
            'gamma': np.arange(1e-3, 1e2),
            #'grow_policy': ['depthwise', 'lossguide'],
            #'min_child_weight': np.arange(1, 10, 1),
            'subsample': np.arange(0.5, 1, 0.1),
            'sampling_method': ['uniform'],
            'colsample_bytree': np.arange(0.5, 1, 0.1),
            'colsample_bylevel': np.arange(0.5, 1, 0.1),
            'colsample_bynode': np.arange(0.5, 1, 0.1),
            'nthread': [8],
            'tree_method': tree_method_list,
            'scale_pos_weight': [self.ratio],
            'predictor': ['cpu_predictor']
            }

        return params

      space = random_params()

      def random_objective(space):

        """
        Defines the objective function to be optimized by the Random Search
        algorithm

        Parameters
        ----------
        space: dict
          A dictionary of parameters to use for each trial of the Random Search
          optimization algorithm

        """
        if not isinstance(space, dict):
          raise TypeError('Parameters must be provided as a dictionary')

        print('Training with params: ')
        print(space)

        results = self.cross_validation(space)
        self.trials = self.trials.append(results, ignore_index=True)

        current_time = (timer()-self.time)/3600
        print('\n')
        print('Time elapsed so far: '+ str(current_time) + ' hours')
        print('\n')

      self.time = timer()
      for i in range(self.max_evals):
        param = {}
        param_sample = {key: random.sample(list(value), 1)[0]
                        for key, value in space.items()}
      #   if param_sample['grow_policy'] == 'lossguide':
      #     param.update({'max_leaves': np.arange(0, 10, 1)})

      #   if param_sample['booster'] == 'dart':
      #     param.update({'sample_type': ['uniform', 'weighted']})
      #     param.update({'normalize_type': ['tree', 'forest']})
      #     param.update({'rate_drop': np.linspace(0, 1)})
      #     param.update({'skip_drop': np.linspace(0, 1)})

      #   if param_sample['tree_method'] == 'hist':
      #     param.update({'max_bin': [2**7, 2**8, 2**9, 2**10]})

      #   if param_sample['tree_method'] == 'approx':
      #     param.update({'sketch_eps': np.arange(0.01, 0.99, 0.01)})

        if param_sample['tree_method'] == 'gpu_hist':
          param.update({'max_bin': [2**7, 2**8, 2**9, 2**10]})
          param.update({'single_precision_histogram': [True, False]})
          param.update({'deterministic_histogram': [True, False]})
          param.update({'subsample': np.arange(0.1, 1, 0.05)})
          param.update({'sampling_method': ['gradient_based']})
          param.update({'predictor': ['gpu_predictor']})

        param_sam = {key: random.sample(list(value), 1)[0]
                     for key, value in param.items()}
        param_sample.update(param_sam)

        random_objective(param_sample)
      self.opt_time = (timer() - self.time)/3600

      if not os.path.exists('XGBoost_trials'):
        os.makedirs('XGBoost_trials')
      self.trials.to_csv('XGBoost_trials/random_search_trials.csv')

      best = self.trials[['params', 'loss']].sort_values(
                                    by='loss', ascending=True).reset_index().loc[0].to_dict()
      self.best_params = best['params']
      print('The best Random Search hyperparameters are: ')
      print(best)
      print('\n')
      return best

    def train(self, optim_type):
      """
      Train the three best models obtained by the HyperOpt, Optuna and Random
      Search optimization algorithms

      Parameters
      ----------
      optim_type: string
        The optimizer type to use. Choice can be hyperopt, optuna, or random_search
      """

      if optim_type not in ['hyperopt', 'optuna', 'random_search']:
        raise ValueError('Optimizer type {} not supported'.format(optim_type))

      print('Starting training...')
      self.output = {}
      self.optim_type = optim_type

      if self.optim_type == 'hyperopt':
        self.hyperopt_tuning()
      elif self.optim_type == 'optuna':
        self.optuna_tuning()
      elif self.optim_type == 'random_search':
#         random.seed(self.seed)
        self.random_search_tuning()

      self.output[self.optim_type] = {}
      self.output[self.optim_type]['params'] = self.best_params
      self.output[self.optim_type]['n_estimators'] = self.estimators
      self.output[self.optim_type]['optimization_time (hours)'] = np.round(self.opt_time, 2)
      
      
      self.best_params['eta'] = 0.05
      start = timer()
      self.best_model = xgb.train(self.best_params, self.dtrain, num_boost_round=self.estimators, feval=f1_eval)
      self.train_time = timer() - start
      self.output[self.optim_type]['train_time (seconds)'] = np.round(self.train_time, 2)
      pickle.dump(self.best_model, open('XGBoost_' + self.optim_type + "_model.dat", "wb"))
      self.trained = True
      self.train_predictions = self.best_model.predict(self.dtrain)

    def test(self, x_test, y_test):
      """
      Use the trained model with the best hyperparameters to predict on the testing set

      Parameters
      ----------
      x_test: numpy array, Pandas dataframe or Series
        Contains observations of predictor features to be used in testing
      y_test: numpy array, Pandas dataframe or Series
        Contains with observations of target feature to be used in testing


      """
      if not self.trained:
        raise Exception('Please train the models using the train_model' +
                                                       'method before testing')
      self.X_test, self.y_test = x_test, y_test
      self.dtest = xgb.DMatrix(x_test, label=self.y_test)
      self.predictions = self.best_model.predict(self.dtest)
      score = log_loss(self.y_test, np.float64(self.predictions))
      self.output[self.optim_type]['score'] = score
        
      self.tested = True
      print('Testing Results: ')
      print(self.output)
      print('\n')

      pd.DataFrame(self.output).to_csv('XGBoost_Summary.csv')

    def feature_importance(self, importance_type='gain'):

      """
      Calculates and plots feature importances of the trained model

      Parameters
      ----------
      importance_type: string, default = 'gain'
        Metric to evaluate feature importance
      """
      if importance_type not in ['weight', 'gain', 'cover',
                                 'total_gain', 'total_cover', 'shap']:
        raise ValueError('Importance Type not supported.Must be among:\'gain',
                        'weight', 'cover', 'total_gain', 'total_cover', 'shap')
      if not self.trained:
        raise Exception('Please train the models using train_models method' +
                                      'before calculating feature importances')

      if importance_type == 'shap':
        if self.best_params['booster'] == 'gbtree':
          explainer = shap.TreeExplainer(self.best_model)
          importance = explainer.shap_values(self.X_train)
          fig = plt.figure()
          fig.suptitle(self.optim_type + ' - Feature Importance - ' + importance_type)
          shap.summary_plot(importance, self.X_train)
        else:
          print('Importance type SHAP only valid for gbtree type booster')
      else:
        importance = self.best_model.get_score(importance_type=importance_type)
        importance = {key: np.round(value, 2)
                      for key, value in importance.items()}
        xgb.plot_importance(importance, importance_type=importance_type)
        plt.title(self.optim_type + ' - Feature Importance - ' + importance_type)
      if not os.path.exists('XGBoost_Plots'):
        os.makedirs('XGBoost_Plots')
      plt.savefig('XGBoost_Plots/XGBoost_' + self.optim_type + '_FeatureImportance.png')
