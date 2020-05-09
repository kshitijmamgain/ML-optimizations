import unittest
import pytest
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal
from unittest import TestCase
import numpy as np
import pandas as pd
from numpy import nan
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import PowerTransformer
import dask
import dask_ml
import dask.dataframe as dd
from dask import compute
from dask_ml.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from dask_ml.decomposition import PCA
from preprocess import Preprocessor

class  TestPreprocess(unittest.TestCase):
    def test_remove_invalid_duplicates(self):
        dask_data = dd.read_csv('data_duplicate.csv')
        x = Preprocessor(['feat1','feat2','feat3'], 'target', dask_data, ['0','1'] , categorical_features = ['feat4'])
        x.execute(duplicates_invalid = True, missing = False, scale = False, transform = False, encode_target = False, train = True)
        expected_output_dict = {'target': {0: '0', 6: '1', 7: '0', 10: '0'},
 'feat1': {0: 1, 6: 3, 7: 2, 10: 5},
 'feat2': {0: 2.0, 6: 3.0, 7: 2.0, 10: 7.0},
 'feat3': {0: 3.0, 6: 4.0, 7: 3.0, 10: 3.0},
 'feat4': {0: 'a', 6: 'a', 7: 'b', 10: 'a'}}

        self.assertEqual(expected_output_dict,x.df.head(8).dropna().to_dict())

    def test_remove_missing_values(self):
        dask_data = dd.read_csv('data_duplicate.csv')
        x = Preprocessor(['feat1','feat2','feat3'], 'target', dask_data, ['0','1'] , categorical_features = ['feat4'])
        x.execute(duplicates_invalid = True, missing = True, scale = False, transform = False, encode_target = False, train = True)
        expected_output_dict = {'target': {0: '0', 1: '1', 2: '0', 6: '1', 7: '0', 8: '1', 9: '1', 10: '0'},
 'feat1': {0: 1, 1: 2, 2: 1, 6: 3, 7: 2, 8: 2, 9: 2, 10: 5},
 'feat2': {0: 2.0, 1: 5.0, 2: 2.0, 6: 3.0, 7: 2.0, 8: 4.0, 9: 3.571, 10: 7.0},
 'feat3': {0: 3.0, 1: 3.2, 2: 3.2, 6: 4.0, 7: 3.0, 8: 3.2, 9: 3.0, 10: 3.0},
 'feat4': {0: 'a', 1: 'Other', 2: 'b', 6: 'a', 7: 'b', 8: 'c', 9: 'c', 10: 'a'}}

        self.assertEqual(expected_output_dict,x.df.round(3).head(8).to_dict())

    def test_scale_data(self):
        dask_data = dd.read_csv('data_duplicate.csv')
        x = Preprocessor(['feat1','feat2','feat3'], 'target', dask_data, ['0','1'] , categorical_features = ['feat4'])
        x.execute(duplicates_invalid = True, missing = True, scale = True, transform = False, encode_target = False, train = True)
        expected_output_dict = {'target': {0: '0', 1: '1', 2: '0', 6: '1', 7: '0', 8: '1', 9: '1', 10: '0'},
 'feat1': {0: -1.043, 1: -0.209, 2: -1.043, 6: 0.626, 7: -0.209, 8: -0.209, 9: -0.209, 10: 2.294},
 'feat2': {0: -0.954, 1: 0.867, 2: -0.954, 6: -0.347, 7: -0.954, 8: 0.26, 9: 0.0, 10: 2.081},
 'feat3': {0: -0.632, 1: 0.0, 2: 0.0, 6: 2.53, 7: -0.632, 8: 0.0, 9: -0.632, 10: -0.632},
 'feat4': {0: 'a', 1: 'Other', 2: 'b', 6: 'a', 7: 'b', 8: 'c', 9: 'c', 10: 'a'}}

        self.assertEqual(expected_output_dict,x.df.round(3).head(8).to_dict())

    def test_transform_date(self):
        dask_data = dd.read_csv('data_duplicate.csv')
        x = Preprocessor(['feat1','feat2','feat3'], 'target', dask_data, ['0','1'] , categorical_features = ['feat4'])
        x.execute(duplicates_invalid = True, missing = True, scale = True, transform = True, encode_target = False, train = True)
        expected_output_dict = {'target': {0: '0', 1: '1', 2: '0', 6: '1', 7: '0', 8: '1', 9: '1', 10: '0'},
 'feat1': {0: -1.043, 1: -0.209, 2: -1.043, 6: 0.626, 7: -0.209, 8: -0.209, 9: -0.209, 10: 2.294},
 'feat2': {0: -0.954, 1: 0.867, 2: -0.954, 6: -0.347, 7: -0.954, 8: 0.26, 9: 0.0, 10: 2.081},
 'feat3': {0: -0.632, 1: 0.0, 2: 0.0, 6: 2.53, 7: -0.632, 8: 0.0, 9: -0.632, 10: -0.632},
 'feat4': {0: 'a', 1: 'Other', 2: 'b', 6: 'a', 7: 'b', 8: 'c', 9: 'c', 10: 'a'}}

        self.assertEqual(expected_output_dict,x.df.round(3).head(8).to_dict())

    def test_encode_and_no_categorical(self):
        dask_data = dd.read_csv('data_encode.csv')
        x = Preprocessor(['feat1','feat2','feat3'], 'target', dask_data, ['o','p','n'] )
        x.execute(duplicates_invalid = True, missing = True, scale = True, transform = True, encode_target = True, train = True)
        expected_output_dict = {'target': {0: 2, 1: 2, 2: 0, 3: 0, 6: 1, 7: 2, 8: 0, 9: 2},
 'feat1': {0: -0.928, 1: -0.093, 2: -0.928, 3: -0.928, 6: 0.743, 7: -0.093, 8: -0.093, 9: -0.093},
 'feat2': {0: -0.844, 1: 0.998, 2: -0.844, 3: -0.844, 6: -0.23, 7: -0.844, 8: 0.384, 9: 0.0},
 'feat3': {0: -0.548, 1: 0.0, 2: 0.0, 3: -0.548, 6: 2.739, 7: -0.548, 8: 0.0, 9: -0.548}}

        self.assertEqual(expected_output_dict,x.df.round(3).head(8).to_dict())






