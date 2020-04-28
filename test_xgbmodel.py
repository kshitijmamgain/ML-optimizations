import unittest
import pytest
import os
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal

from xgbmodel import XGBoostModel
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
import random
import seaborn as sns
import shap
from unittest import TestCase

class TestXGB(unittest.TestCase):
    def test_train_models_on_normal_cases (self):
        x = XGBoostModel('data.csv','infer',10,2,10,2,42,False)
        x.train_models()
        expected_train_hyperopt_params = {'booster': 'gbtree',
 'colsample_bylevel': 0.55,
 'colsample_bynode': 1.0,
 'colsample_bytree': 0.4,
 'disable_default_eval_metric': 1,
 'eta': 0.17500000000000002,
 'eval_metric': 'logloss',
 'gamma': 0.6000000000000001,
 'grow_policy': 'depthwise',
 'max_delta_step': 3.0,
 'max_depth': 9,
 'min_child_weight': 4.0,
 'objective': 'binary:logistic',
 'predictor': 'cpu_predictor',
 'reg_alpha': 5.0,
 'reg_lambda': 1.3,
 'sampling_method': 'uniform',
 'scale_pos_weight': 0.8621973929236499,
 'subsample': 0.55,
 'tree_method': 'approx',
 'verbosity': 1,
 'sketch_eps': 0.05,
 'n_estimators': 10,
 'validate_parameters': 1}

        expected_train_optuna_params = {'objective': 'binary:logistic',
 'eval_metric': 'logloss',
 'verbosity': 1,
 'disable_default_eval_metric': 1,
 'booster': 'gbtree',
 'reg_lambda': 2,
 'reg_alpha': 10,
 'max_delta_step': 8,
 'max_depth': 13,
 'eta': 0.30850382502458135,
 'gamma': 0.7229163764267956,
 'subsample': 0.5499874579090014,
 'grow_policy': 'depthwise',
 'min_child_weight': 8.795585311974417,
 'colsample_bytree': 0.6410035105688879,
 'colsample_bylevel': 0.737265320016441,
 'colsample_bynode': 0.1185260448662222,
 'tree_method': 'exact',
 'scale_pos_weight': 0.8621973929236499,
 'sampling_method': 'uniform',
 'predictor': 'cpu_predictor',
 'n_estimators': 9,
 'validate_parameters': 1}

        expected_train_random_search_params = {'objective': 'binary:logistic',
 'eval_metric': 'logloss',
 'disable_default_eval_metric': 1,
 'booster': 'gbtree',
 'reg_lambda': 1.3000000000000003,
 'reg_alpha': 2,
 'verbosity': 1,
 'max_delta_step': 9,
 'max_depth': 2,
 'eta': 0.47500000000000003,
 'gamma': 0.8000000000000003,
 'grow_policy': 'depthwise',
 'min_child_weight': 1,
 'subsample': 0.55,
 'sampling_method': 'uniform',
 'colsample_bytree': 0.45000000000000007,
 'colsample_bylevel': 0.9000000000000002,
 'colsample_bynode': 0.1,
 'tree_method': 'exact',
 'scale_pos_weight': 0.8621973929236499,
 'predictor': 'cpu_predictor',
 'n_estimators': 4,
 'validate_parameters': 1}
                        
        self.assertEqual(expected_train_hyperopt_params,x.best_hyperopt_params)
        self.assertEqual(expected_train_optuna_params, x.best_optuna_params)
        self.assertEqual(expected_train_random_search_params, x.best_random_search_params)
        assert x.best_hyperopt_model == None
        assert x.best_optuna_model == None
        assert x.best_random_search_model == None



    def test_test_models_on_normal_cases (self):
        x = XGBoostModel('data.csv','infer',10,2,10,2,42,False)
        x.train_models()
        x.test_models()
        expected_hyperopt_predict = [0.441, 0.613, 0.507, 0.376, 0.6, 0.528, 0.536, 0.482, 0.48, 0.596]
        expected_hyperopt_score = 0.68
        expected_optuna_predict = [0.555, 0.513, 0.5, 0.369, 0.531, 0.538, 0.476, 0.505, 0.489, 0.562]
        expected_optuna_score = 0.68
        expected_random_search_predict = [0.479, 0.523, 0.507, 0.34, 0.551, 0.454, 0.478, 0.493, 0.596, 0.523]
        expected_random_search_score = 0.69

        hyperopt_pred = [(lambda y:round(y,3))(val) for val in x.output['hyperopt']['prediction'][0:10]]
        optuna_pred = [(lambda y:round(y,3))(val) for val in x.output['optuna']['prediction'][0:10]]
        random_search_pred = [(lambda y:round(y,3))(val) for val in x.output['random_search']['prediction'][0:10]]
        
        self.assertEqual(expected_hyperopt_predict, hyperopt_pred)
        self.assertEqual(expected_hyperopt_score, round(x.output['hyperopt']['score'],2))
        self.assertEqual(expected_optuna_predict, optuna_pred)
        self.assertEqual(expected_optuna_score, round(x.output['optuna']['score'],2))
        self.assertEqual(expected_random_search_predict, random_search_pred)
        self.assertEqual(expected_random_search_score, round(x.output['random_search']['score'],2))
        assert os.path.exists('XGBoost_Summary.csv')


    def test_test_models_on_bad_case (self):
        with pytest.raises(Exception) as exc_info:
            x = XGBoostModel('data.csv','infer',10,2,10,2,42,False)
            x.test_models()

        assert exc_info.match('Please train the models using train_models method \
before testing')

    def test_featuere_importance_shap_on_normal_case(self):
        x = XGBoostModel('data.csv','infer',10,2,10,2,42,False)
        x.train_models()
        x.test_models()
        x.feature_importance('shap')

        expected_list = [0.0, 0.018, -0.008, 0.07, 0.0, 0.022, 0.0, 0.0, -0.014, 0.0,
         0.025, -0.009, 0.0, -0.031, 0.0, 0.097, 0.0, 0.0, 0.0, 0.022, 0.0]
        shap_values = [(lambda y:round(y,3))(val) for val in x.importance[:1,:21][0]]

        assert_almost_equal(expected_list, shap_values)
        assert os.path.exists('XGBoost_Plots/XGBoost_hyperopt_FeatureImportance.png')
        assert os.path.exists('XGBoost_Plots/XGBoost_optuna_FeatureImportance.png')
        assert os.path.exists('XGBoost_Plots/XGBoost_random_search_FeatureImportance.png')

    
    def test_featuere_importance_non_shap_on_normal_case(self):
        x = XGBoostModel('data.csv','infer',10,2,10,2,42,False)
        x.train_models()
        x.test_models()
        x.feature_importance()
        expected_imp = {'4': 23.96,
 '12': 3.64,
 '3': 3.69,
 '6': 12.21,
 '2': 3.74,
 '14': 3.32,
 '11': 2.69,
 '9': 1.03,
 '16': 3.53,
 '20': 1.82}

        self.assertDictEqual(expected_imp, x.importance)
        assert os.path.exists('XGBoost_Plots/XGBoost_hyperopt_FeatureImportance.png')
        assert os.path.exists('XGBoost_Plots/XGBoost_optuna_FeatureImportance.png')
        assert os.path.exists('XGBoost_Plots/XGBoost_random_search_FeatureImportance.png')

    def test_confusion_normal_cases(self):
        x = XGBoostModel('data.csv','infer',10,2,10,2,42,False)
        x.train_models()
        x.test_models()
        x.confusion_matrix()
        expected_cm = [[604, 811], [529, 1056]]

        self.assertEqual(expected_cm, x.cm.tolist())

    def test_roc_curve_normal_cases(self):
        x = XGBoostModel('data.csv','infer',10,2,10,2,42,False)
        x.train_models()
        x.test_models()
        x.roc_curve()
        expected_fpr = [0.0, 0.0, 0.0, 0.001, 0.002, 0.003, 0.003, 0.004, 0.004, 0.006]
        expected_tpr = [0.0, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.005, 0.006, 0.006]
        expected_thresholds = [1.726, 0.726, 0.725, 0.704, 0.703, 0.697, 0.695, 0.674, 0.673, 0.666]
        fpr_result = [(lambda y:round(y,3))(val) for val in x.fpr[0:10]]
        tpr_result = [(lambda y:round(y,3))(val) for val in x.tpr[0:10]]
        thresholds_result = [(lambda y:round(y,3))(val) for val in x.thresholds[0:10]]

        self.assertEqual(expected_fpr, fpr_result)
        self.assertEqual(expected_tpr, tpr_result)
        self.assertEqual(expected_thresholds, thresholds_result)
        assert os.path.exists('XGBoost_Plots/XGBoost_hyperopt_ROC_Curve.png')
        assert os.path.exists('XGBoost_Plots/XGBoost_optuna_ROC_Curve.png')
        assert os.path.exists('XGBoost_Plots/XGBoost_random_search_ROC_Curve.png')

    def test_pr_curve_normal_cases(self):
        x = XGBoostModel('data.csv','infer',10,2,10,2,42,False)
        x.train_models()
        x.test_models()
        x.pr_curve()
        expected_precision = [0.53, 0.531, 0.531, 0.531, 0.531, 0.531, 0.531, 0.531, 0.531, 0.532]
        expected_recall = [1.0, 0.998, 0.997, 0.997, 0.997, 0.997, 0.997, 0.997, 0.997, 0.997]
        precision_result = [(lambda y:round(y,3))(val) for val in x.precision[0:10]]
        recall_result = [(lambda y:round(y,3))(val) for val in x.recall[0:10]]

        self.assertEqual(expected_precision, precision_result)
        self.assertEqual(expected_recall, recall_result)
        assert os.path.exists('XGBoost_Plots/XGBoost_hyperopt_PR_Curve.png')
        assert os.path.exists('XGBoost_Plots/XGBoost_optuna_PR_Curve.png')
        assert os.path.exists('XGBoost_Plots/XGBoost_random_search_PR_Curve.png')
