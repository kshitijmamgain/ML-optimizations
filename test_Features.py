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
        
        x=FeatureEng('file_1.csv','infer' )
        x.separate_features_from_label('target')
        a=[]
        b=['add_numeric', 'multiply_numeric']
        feat_matrix,feat_defs = x.new_features(a,b,2)
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
            x=FeatureEng("file_2.csv",'infer')
            x.separate_features_from_label('target')
            a=[]
            b=['add_numeric', 'multiply_numeric']
            feat_matrix,feat_defs = x.new_features(a,b,2)

        assert exc_info.match("Data Frame contains non-numeric values")

class TestFeaturesTwo(unittest.TestCase):
    def test_df_with_new_features_on_normal_values(self):
        x=FeatureEng('file_1.csv','infer')
        x.separate_features_from_label('target')
        a=[]
        b=['add_numeric', 'multiply_numeric']
        feat_matrix,feat_defs = x.new_features(a,b,2)
        feat_matrix = x.df_with_new_features('target')
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
            x=FeatureEng('file_1.csv','infer' )
            x.separate_features_from_label('feat2')
            a=[]
            b=['add_numeric', 'multiply_numeric']
            feat_matrix,feat_defs = x.new_features(a,b,2)
            feat_matrix = x.df_with_new_features('target')

        assert exc_info_2.match("Data Frame already contains target column")

    
class TestFeatureSelect(unittest.TestCase):

    def test_combine_selector_on_normal_cases(self):
        
        x = FeatureEng('file_1.csv','infer')
        x.separate_features_from_label('target')
        a=[]
        b=['add_numeric', 'multiply_numeric']
        feat_matrix,feat_defs = x.new_features(a,b,1)
        feat_matrix=x.df_with_new_features('target')

        x_2 = FeatureSelect(x.feature_matrix, 6)
        x_2.cor_pearson_selector()
        x_2.chi_square_selector()
        x_2.recursive_selector()
        x_2.log_reg_selector()
        x_2.random_forest_selector()
        x_2.LGBM_selector()
        x_2.Extra_Trees_selector()
        x_2.combine_selector()

        #define the correct result dataframe that we expect to get
        values = [['feat1 + feat3',True,True,True,True,True,True,True,7],
            ['feat3',True,True,True,True,False,True,True,6],
            ['feat1 * feat3',True,True,True,True,True,False,True,6],
            ['feat1 + feat2',True,True,True,True,False,True,False,5],
            ['feat1 * feat2',True,True,True,False,True,False,True,5],
            ['feat2 * feat3',True,True,True,True,False,False,False,4]
           ]
        col_names = ['Feature', 'Pearson', 'Chi-2', 'RFE', 'Logistics', 'Random Forest','LightGBM','Extra_trees','Total']
        df_result = pd.DataFrame(data = values, columns = col_names)
        df_result.index += 1 

        assert_frame_equal(df_result,x_2.combine_selector())

    def test_df_selected_columns_normal_case(self):
        x = FeatureEng('file_1.csv','infer')
        x.separate_features_from_label('target')
        a=[]
        b=['add_numeric', 'multiply_numeric']
        feat_matrix,feat_defs = x.new_features(a,b,1)
        feat_matrix = x.remove_correlated_features(0.8)
        feat_matrix=x.df_with_new_features('target')

        x_2 = FeatureSelect(x.feature_matrix, 3)
        x_2.cor_pearson_selector()
        x_2.chi_square_selector()
        x_2.recursive_selector()
        x_2.log_reg_selector()
        x_2.random_forest_selector()
        x_2.LGBM_selector()
        x_2.Extra_Trees_selector()
        x_2.combine_selector()
        x_2.save_df_selected_columns()

        #define the correct result dataframe that we expect to get
        values = [[1,2,5,0],
            [1,5,5,1],
           ]
        col_names = ['feat1','feat2','feat2 + feat3','target']
        df_result = pd.DataFrame(data = values, columns = col_names)
        #df_result.index += 0
        x_2.df_selected_columns.reset_index(drop=True, inplace=True)
        df_result.reset_index(drop=True, inplace=True)

        assert_frame_equal(df_result,x_2.df_selected_columns)
        




