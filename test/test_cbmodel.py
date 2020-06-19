import unittest
import pytest
import os
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal
import optuna
import csv
from timeit import default_timer as timer
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (auc, accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, precision_recall_curve, 
                             classification_report)
from hyperopt import STATUS_OK, STATUS_FAIL, hp, tpe, Trials, fmin
import matplotlib.pyplot as plt
import catboost as cb
from catboost import CatBoost
import shap
from google.colab import files
from unittest import TestCase
import cbmodel

class TestCBoost(unittest.TestCase):
    def test_train_hyperopt_on_normal_cases (self):
        df = pd.read_csv('data.csv', header = 'infer')
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        obj= cbmodel.Ctbclass(x_train, y_train, 'GPU', 'hyperopt', 10, 10, 2)
        obj.train(x_test,y_test)
        expected_best_params = {'bootstrap_type': 'Bernoulli',
 'border_count': 60,
 'custom_loss': 'TotalF1',
 'depth': 4,
 'eval_metric': 'AUC',
 'grow_policy': 'Lossguide',
 'l2_leaf_reg': 1,
 'leaf_estimation_backtracking': 'AnyImprovement',
 'learning_rate': 0.41996712228573996,
 'loss_function': 'Logloss',
 'max_leaves': 29.0,
 'min_data_in_leaf': 22.0,
 'n_estimators': 9,
 'od_type': 'IncToDec',
 'score_function': 'LOOL2',
 'task_type': 'GPU',
 'thread_count': 2}

        expected_cv_results = {'iterations': {9: 9},
 'test-AUC-mean': {9: 0.748},
 'test-AUC-std': {9: 0.009},
 'test-CrossEntropy-mean': {9: 0.595},
 'test-CrossEntropy-std': {9: 0.009},
 'test-Logloss-mean': {9: 0.595},
 'test-Logloss-std': {9: 0.009},
 'train-CrossEntropy-mean': {9: 0.461},
 'train-CrossEntropy-std': {9: 0.001},
 'train-Logloss-mean': {9: 0.461},
 'train-Logloss-std': {9: 0.001}}

        expected_best_score = 0.749

        self.assertEqual(expected_best_params,obj.params)
        self.assertEqual(expected_cv_results,round(obj.cv_results,3).iloc[9:].to_dict())
        self.assertEqual(expected_best_score,round(obj.best_score,3))

    def test_train_optuna_on_normal_cases (self):
        df = pd.read_csv('data.csv', header = 'infer')
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        obj= cbmodel.Ctbclass(x_train, y_train, 'GPU', 'optuna', 10, 10, 2)
        obj.train(x_test,y_test)
        expected_best_params = {'bootstrap_type': 'Bernoulli',
 'border_count': 82,
 'depth': 6,
 'eval_metric': 'AUC',
 'grow_policy': 'Lossguide',
 'l2_leaf_reg': 2,
 'leaf_estimation_backtracking': 'Armijo',
 'learning_rate': 0.17421929242173662,
 'loss_function': 'Logloss',
 'min_data_in_leaf': 25,
 'n_estimators': 10,
 'od_type': 'IncToDec',
 'score_function': 'L2',
 'task_type': 'GPU'}

        expected_cv_results = {'iterations': {9: 9},
 'test-AUC-mean': {9: 0.713},
 'test-AUC-std': {9: 0.015},
 'test-CrossEntropy-mean': {9: 0.628},
 'test-CrossEntropy-std': {9: 0.006},
 'train-CrossEntropy-mean': {9: 0.621},
 'train-CrossEntropy-std': {9: 0.003}}

        expected_best_score = 0.713

        self.assertEqual(expected_best_params,obj.params)
        self.assertEqual(expected_cv_results,round(obj.cv_results,3).iloc[9:].to_dict())
        self.assertEqual(expected_best_score,round(obj.best_score,3))