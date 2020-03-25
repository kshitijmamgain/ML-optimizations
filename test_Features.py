import pandas as pd
import pytest
from Features import FeatureEng
import unittest


class TestFeatures(unittest.TestCase):
    def test_new_features_on_normal_values (self):
        
        x=FeatureEng('file_1.csv','infer' )
        x.separate_features_from_label('target')
        a=[]
        b=['add_numeric', 'multiply_numeric']
        feat_matrix,feat_defs = x.new_features(a,b)
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
            feat_matrix,feat_defs = x.new_features(a,b)

        assert exc_info.match("Data Frame contains non-numeric values")

class TestFeaturesTwo(unittest.TestCase):
    def test_df_with_new_features_on_normal_values(self):
        x=FeatureEng('file_1.csv','infer')
        x.separate_features_from_label('target')
        a=[]
        b=['add_numeric', 'multiply_numeric']
        feat_matrix,feat_defs = x.new_features(a,b)
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
            feat_matrix,feat_defs = x.new_features(a,b)
            feat_matrix = x.df_with_new_features('target')

        assert exc_info_2.match("Data Frame already contains target column")
