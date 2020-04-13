import pytest
from Features import FeatureEng
from Features import FeatureSelect
import unittest
from pandas.testing import assert_frame_equal

import pandas as pd
import numpy as np
import featuretools as ft
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFromModel


class TestFeatures(unittest.TestCase):
    def test_new_features_on_normal_values (self):
        
        a=[]
        b=['add_numeric', 'multiply_numeric']
        x=FeatureEng('file_1.csv','infer', None,'target',a,b,2,0.8 )
        feat_matrix = x.new_features()
        message = "column names do not match"
        correct_columns = ['feat1',
 'feat2',
 'feat3',
 'feat1 + feat3',
 'feat1 + feat2',
 'feat2 + feat3',
 'feat1 * feat3',
 'feat2 * feat3',
 'feat1 * feat2',
 'feat1 + feat2 * feat2 + feat3',
 'feat1 + feat3 * feat2 + feat3',
 'feat1 + feat2 * feat3',
 'feat1 + feat2 * feat1 + feat3',
 'feat1 + feat2 * feat2',
 'feat1 * feat2 + feat3',
 'feat1 * feat1 + feat2',
 'feat1 + feat3 * feat2',
 'feat1 + feat3 * feat3',
 'feat2 + feat3 * feat3',
 'feat1 * feat1 + feat3',
 'feat2 * feat2 + feat3']
        assert feat_matrix.columns.isin(correct_columns).all(),message
        assert not ~ feat_matrix.columns.isin(correct_columns).all(), message

    def test_new_features_on_containing_non_numeric_values(self):
        with pytest.raises(ValueError) as exc_info:
            a=[]
            b=['add_numeric', 'multiply_numeric']
            x=FeatureEng("file_2.csv",'infer', None,'target',a,b,2,0.8)
            x.new_features()

        assert exc_info.match("Data Frame contains non-numeric values")

class TestFeaturesTwo(unittest.TestCase):
    def test_df_with_new_features_on_normal_values(self):
        a=[]
        b=['add_numeric', 'multiply_numeric']
        x=FeatureEng('file_1.csv','infer', None,'target',a,b,2,0.8)
        feat_matrix = x.df_with_new_features()
        message = "column names do not match"
        correct_columns = ['feat1',
 'feat2',
 'feat3',
 'feat1 + feat3',
 'feat1 + feat2',
 'feat2 + feat3',
 'feat1 * feat3',
 'feat2 * feat3',
 'feat1 * feat2',
 'feat1 + feat2 * feat2 + feat3',
 'feat1 + feat3 * feat2 + feat3',
 'feat1 + feat2 * feat3',
 'feat1 + feat2 * feat1 + feat3',
 'feat1 + feat2 * feat2',
 'feat1 * feat2 + feat3',
 'feat1 * feat1 + feat2',
 'feat1 + feat3 * feat2',
 'feat1 + feat3 * feat3',
 'feat2 + feat3 * feat3',
 'feat1 * feat1 + feat3',
 'feat2 * feat2 + feat3',
 'target']
        assert feat_matrix.columns.isin(correct_columns).all(), message
        assert not ~ feat_matrix.columns.isin(correct_columns).all(), message


    def test_df_with_new_features_on_containing_target_column(self):
        #test when target column exists in dataset (it was not removed correctly by separate_features_from_label stage)
        with pytest.raises(ValueError) as exc_info_2:
            a=[]
            b=['add_numeric', 'multiply_numeric']
            x=FeatureEng('file_1.csv','infer',None,'feat1',a,b,2,0.8 )
            x.df_with_new_features()

        assert exc_info_2.match("Data Frame already contains target column")

    
class TestFeatureSelect(unittest.TestCase):

    def test_combine_selector_on_normal_cases(self):
       
        a=[]
        b=['add_numeric', 'multiply_numeric']
        x = FeatureEng('file_1.csv','infer',None,'target',a,b,1,0)
        x.new_features()

        x_2 = FeatureSelect(x.feature_matrix, 6)
        x_2.combine_selector()

        #define the correct result dataframe that we expect to get
        values = [['feat3',True,True,True,True,True,True,6],
            ['feat2',True,True,True,True,True,True,6],
            ['feat1 + feat3',True,True,True,True,True,True,6],
            ['feat1 + feat2',True,True,True,True,True,True,6],
            ['feat2 * feat3',True,True,True,False,False,False,3],
            ['feat1 * feat3',True,True,True,False,False,False,3]
           ]
        col_names = ['Feature', 'Pearson', 'Chi-2', 'Logistics', 'Random Forest','LightGBM','Extra_trees','Total']
        df_result = pd.DataFrame(data = values, columns = col_names)
        df_result.index += 1 

        assert_frame_equal(df_result,x_2.combine_selector())

    def test_df_selected_columns_normal_case(self):
        
        a=[]
        b=['add_numeric', 'multiply_numeric']
        x = FeatureEng('file_1.csv','infer',None,'target',a,b,4,0.5)
        x.df_with_new_features()

        x_2 = FeatureSelect(x.df_new_features, 3)
        x_2.save_df_selected_columns()

        #define the correct result dataframe that we expect to get
        values = [[2,5,5,0],
            [5,5,5,1],
           ]
        col_names = ['feat2','feat2 + feat3','feat1 * feat2 + feat3','target']
        df_result = pd.DataFrame(data = values, columns = col_names)
        #df_result.index += 0
        x_2.df_selected_columns.reset_index(drop=True, inplace=True)
        df_result.reset_index(drop=True, inplace=True)

        assert_frame_equal(df_result,x_2.df_selected_columns)
        




