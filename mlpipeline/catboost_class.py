import random
from timeit import default_timer as timer
import csv
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (auc, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, 
                             precision_recall_curve, classification_report)
from hyperopt import STATUS_OK, STATUS_FAIL, hp, tpe, Trials, fmin
import matplotlib.pyplot as plt
import catboost as cb
from catboost import CatBoost
import shap
#from google.colab import files

df = pd.read_csv('c:/users/hajyhass/downloads/sample.csv', header=None)
X = df.iloc[:, 2:]
y = df.iloc[:, 1]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.0005)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.666)

#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 5 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round
MAX_EVALS = 5

#CATBOOST PARAMETERS
CB_MAX_DEPTH = 16 #maximum tree depth in CatBoost
OBJECTIVE_CB_REG = 'MAE' #CatBoost regression metric
OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric
NUM_BOOST_ROUNDS = 100
EARLY_STOPPING_ROUNDS = 25
SEED = 47

# Hyperopt Space
H_SPACE = {
    'l2_leaf_reg': hp.qloguniform('l2_leaf_reg', 0, 2, 1),
    'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
    'depth': hp.quniform('depth', 1, CB_MAX_DEPTH, 1),
    'loss_function': hp.choice('loss_function', ['Logloss']),#RMSE and #MAE, Poisson for regression
    'border_count': hp.quniform('border_count', 32, 255, 1),
    'bootstrap_type': hp.choice('bootstrap_type', 
                                [{'bootstrap_type': 'Bayesian',
                                  'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
                                 {'bootstrap_type': 'Bernoulli'}]),
    'grow_policy': hp.choice('grow_policy',
                             [{'grow_policy': 'SymmetricTree'}, {'grow_policy': 'Depthwise'},
                              {'grow_policy': 'Lossguide', 
                               'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]),
    'score_function': hp.choice('score_function', ['Cosine']),
    # The score type used to select the next split during the tree construction
    # 'score_function'=['Correlation','L2','NewtonCorrelation','NewtonL2','LOOL2','Cosine','SatL2','SolarL2']
    # NewtonL2 is the only possible choice for Lossguide
    # The other choices explicitely belong to GPU, only 'Cosine' applies to CPU
    'eval_metric': hp.choice('eval_metric', ['F1']),
    # Eval_metric helps with detecting overfitting
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 50, 1),
    'random_strength': hp.loguniform('random_strength', np.log(0.005), np.log(5)),
    'rsm': hp.uniform('rsm', 0.1, 1),
    # RSM :Random subspace method. The percentage of features to use at each split selection
    'od_type': hp.choice('od_type', ['IncToDec', 'Iter']),
    'task_type': 'CPU',
    # 'thread_count':-1
    'leaf_estimation_backtracking': hp.choice('leaf_estimation_backtracking', ['No', 'AnyImprovement']),
    #'boosting_type':hp.choice('boosting_type',['Ordered','Plain']),depends on symmetry of grow_policy
    #Ordered for small requiring high accuracy, Plain for large data requiring processing speed
    }


class Ctbclass():
    '''Catboost Class applying Hyperopt and Optuna techniques '''
    iteration = 0
    def __init__(self, x_train, y_train):
        '''Initializes Catboost Train dataset object
        Parameters
        ----------
        x_train: train data
        y_train: label data'''

        self.x_train = x_train
        self.y_train = y_train
        self.train_set = cb.Pool(self.x_train, self.y_train)
        
    def ctb_crossval(self, params, optim_type):
        '''catboost cross validation model
        Paramters
        ---------
        params: Hyper parameters in dict type from different optimization methods
        optim_type: choose among Optuna, Hyperopt, RandomSearch
        Returns
        ------
        Loss, params, n_estimator, run_time'''
        # initializing the timer
         
        start = timer()
        # if optim_type == 'Optuna':
        #     cv_results = cb.cv(self.train_set, params, fold_count=N_FOLDS, num_boost_round=NUM_BOOST_ROUNDS,
        #                    early_stopping_rounds=EARLY_STOPPING_ROUNDS, stratified=True, partition_random_seed=SEED,
        #                    plot=True)
        # else:
        cv_results = cb.cv(self.train_set, params, fold_count=N_FOLDS, 
                           num_boost_round=NUM_BOOST_ROUNDS,
                           early_stopping_rounds=EARLY_STOPPING_ROUNDS, stratified=True, 
                           partition_random_seed=SEED, plot=True)
        # store the runtime
        run_time = timer() - start

        # Extract the best score
        best_score = np.max(cv_results['test-F1-mean'])

        # Loss must be minimized
        loss = 1 - best_score

        # Boosting rounds that returned the highest cv score
        n_estimators = int(np.argmax(cv_results['test-F1-mean']) + 1)
        self.estimator = n_estimators
        return loss, params, n_estimators, run_time
    def parameter_tuning(self, tune_type):
        tuner = getattr(self, tune_type)
        return tuner()
    
    def hyperopt_space(self):
        '''A method to call the hyperopt optimization
        Parameters
        ----------
        fn_name: is the objective function to minimize defined within the class function
        space: is the hypeorpt space provided as dictionary 
        algo: is the type of search algorithm
        trials: Hyperopt base trials object
        Returns
        -------
        result: best parameter that minimizes the fn_name over max_evals = MAX_EVALS 
        trials: the database in which to store all the point evaluations of the search'''
        fn_name, space, algo, trials = 'hyperopt_obj', H_SPACE, tpe.suggest, Trials()
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=MAX_EVALS,
                          trials=trials, rstate=np.random.RandomState(SEED))
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}
        self.params = trials.best_trial['result']['params']
        self.params['n_estimators'] = self.estimator
        #print(result,trials)
        return result, trials
    def optuna_space(self):
        '''A method to call the optuna optimization
        Parameters
        ----------
        fn_name: is the objective function to minimize defined within the class function
        Returns
        -------
        study: best parameter that minimizes the fn_name over max_evals = MAX_EVALS'''
        fn_name = 'optuna_obj'
        fn = getattr(self, fn_name)
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(fn, n_trials=MAX_EVALS)
        except Exception as e:
            return {'exception': str(e)}
        self.params = study.best_params
        self.params['n_estimators'] = self.estimator
        return study
    def hyperopt_obj(self, params):
        """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

        optim_type = 'Hyperopt'
        self.iteration += 1
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
            # when 'grow_policy' is 'lossguide',the only possible score_function is 'NewtonL2'
        else:
            params['grow_policy'] = params['grow_policy']['grow_policy']
            print(params['grow_policy'])

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['l2_leaf_reg', 'depth', 'border_count']:
            params[parameter_name] = int(params[parameter_name])
        
        
        # Perform n_folds cross validation
        loss, params, n_estimators, run_time = self.ctb_crossval(params, optim_type)

        # Dictionary with information for evaluation
        return {'loss':loss, 'params':params, 'iteration':self.iteration,
                'estimators':n_estimators, 'train_time':run_time, 'status':STATUS_OK}
    def optuna_obj(self, trial):
        """Objective function for Gradient Boosting Machine Optuna Optimization"""
        params = {
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 0, 2, 1),
            'learning_rate': trial.suggest_uniform('learning_rate', 1e-3, 5e-1),
            'depth': trial.suggest_int('depth', 1, CB_MAX_DEPTH, 1),
            #'loss_function': trial.suggest_categorical('loss_function', ['Logloss', None]),
            'loss_function': trial.suggest_categorical('loss_function',
                                                       ['Logloss', 'CrossEntropy']),
            'border_count': trial.suggest_int('border_count', 32, 255, 1),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', 
                                                        ['Bayesian', 'Bernoulli']), 
            'grow_policy': trial.suggest_categorical('grow_policy',
                                                     ['SymmetricTree', 'Depthwise', 'Lossguide']),
            #'score_function': trial.suggest_categorical('score_function', [None, 'Cosine']),
            'score_function': 'Cosine',
            'eval_metric': 'F1',
            #'eval_metric': trial.suggest_categorical('eval_metric', ['F1', 'AUC']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50, 1),
            'random_strength': trial.suggest_uniform('random_strength',
                                                     np.log(0.005), np.log(5)),
            'rsm': trial.suggest_uniform('rsm', 0.1, 1),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
            'task_type': 'CPU',
            'leaf_estimation_backtracking': trial.suggest_categorical('leaf_estimation_backtracking',
                                                                      ['No', 'AnyImprovement'])
        }

        optim_type = 'Optuna'
        self.iteration += 1

        # Perform n_folds cross validation
        if params['grow_policy'] == 'lossguide':
            params['max_leaves'] = trial.suggest_int('max_leaves', 2, 32)
            params['score_function'] = 'NewtonL2'
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_uniform('bagging_temperature',
                                                                  np.log(1), np.log(50))
            
        #for parameter_name in ['l2_leaf_reg', 'depth', 'border_count']:
         #   params[parameter_name] = int(params[parameter_name])
        

        loss, params, _, _ = self.ctb_crossval(params, optim_type)

        return loss
    
    
    def train(self, x_test, y_test):
        """This function trains the globally optimal model on the test data
        Parameters
        ----------
        x_test: test set
        y_test: test label"""
        
        for parameter_name in ['l2_leaf_reg', 'depth', 'border_count']:
            self.params[parameter_name] = int(self.params[parameter_name])

        self.test_set = cb.Pool(x_test, y_test)
        self.cat = cb.train(params=self.params, pool=self.train_set)
        self.pred = self.cat.predict(x_test, prediction_type="Class")
        
        self.test_y = y_test
        self.test_x = x_test
        print("Model will be trained with best parameters obtained ... \n\n\n")
        print("Model trained with {} estimators on the following parameters: \n{}"
              .format(self.estimator, self.params))

    def shap_summary(self):
        """This function plots the Shap values in order of significance"""
        x_test = self.test_x
        z = shap.sample(x_test, nsamples=100)
        explainer = shap.KernelExplainer(self.cat.predict, z)
        k_shap_values = explainer.shap_values(x_test)
        print("Shap Summary Plot")
        plt.figure()
        shap.summary_plot(k_shap_values, x_test, show=False)
        plt.savefig('shap_summary.png')
        
    def shap_collective(self):
        """This function plots the interactive Shap values"""
        shap.initjs()
        x_test = self.test_x
        z = shap.sample(x_test, nsamples=100)
        explainer = shap.KernelExplainer(self.cat.predict, z)
        k_shap_values = explainer.shap_values(x_test)
        return shap.force_plot(explainer.expected_value, k_shap_values, x_test)
     
    def performance(self):
        """This function generates the classification report,
        and confusion matrix"""
        y_test = self.test_y
        y_test = np.array(y_test)
        predictions = self.pred
        # Confusion matrix
        print(confusion_matrix(y_test, predictions))
        # Accuracy, Precision, Recall, F1 score
        print(classification_report(y_test, predictions))


    def evaluate(self):
        """This function generates the evaluation metrics,
        which will later be used for the standard performance plots
        e.g. roc, prcurve, fpr_fnr """
        pred = self.pred
        print('check pred')
        (self.fpr, self.tpr, self.thresholds) = roc_curve(y_true=self.test_y, y_score=pred)
        print('fpr, tpr, thresh check')
        self.fnr = 1- self.tpr
        print('fnr check')
        self.roc_auc = auc(self.fpr, self.tpr)
        print('roc_Auc check')
        self.precision, self.recall, _ = precision_recall_curve(self.test_y, pred)
        print('precision recall check')
        self.pr_auc = auc(self.recall, self.precision)
        print('pr_auc check')
        eval_list = ['roc', 'prcurve', 'fpr_fnr']
        for eval_name in eval_list:
            func = getattr(self, eval_name)
            func()
        else:
            print('Not valid evaluation type')
        
    def roc(self):
        fpr, tpr, roc = self.fpr, self.tpr, self.roc_auc
        plt.figure(figsize=(16, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % self.roc_auc, alpha=0.5)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('Receiver operating characteristic', fontsize=20)
        plt.legend(loc="lower right", fontsize=16)
        plt.savefig('roc.png')
    def prcurve(self):
        test_y = self.test_y
        recall, precision, pr_auc = self.recall, self.precision, self.pr_auc
        # plot the precision-recall curves
        no_skill = len(test_y[test_y == 1])/len(test_y)
        plt.figure(figsize=(16, 8))
        plt.plot([0, 1], [no_skill, no_skill], color='navy', linestyle='--',
                 alpha=0.5)
        plt.plot(recall, precision, color='darkorange',
                 label='ROC curve (area = %0.2f)'% pr_auc, alpha=0.5)
        # axis labels
        plt.title('Precision Recall Curve', size=20)
        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # show the legend
        plt.legend(fontsize=16)
        plt.savefig('prcurve.png')
    def fpr_fnr(self):
        lw = 2
        fpr, fnr, thresholds = self.fpr, self.fnr, self.thresholds
        plt.figure(figsize=(16, 8))
        plt.plot(thresholds, fpr, color='blue', lw=lw, label='FPR', alpha=0.5)
        plt.plot(thresholds, fnr, color='green', lw=lw, label='FNR', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('Threshold', fontsize=16)
        plt.ylabel('Error Rate', fontsize=16)
        plt.title('FPR-FNR curves', fontsize=20)
        plt.legend(loc="lower left", fontsize=16)
        plt.savefig('fpr-fnr.png')
        
#obj= Ctbclass(x_train, y_train)
# obj.optuna_space()
# #obj.hyperopt_space()
# obj.train(x_train,y_train)
# obj.evaluate()
