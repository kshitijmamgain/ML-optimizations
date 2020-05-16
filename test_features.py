import pytest
from features import FeatureEng
from features import FeatureSelect
import featuretools as ft

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class TestFeature():
    '''Tests FEatureEng and FeatureSelect class in Features.py'''
    def test_show_features(self):
        '''Tests the show and separate_features_from_label methods in FeatureEng class'''

        cat_col_name = []
        ap = []
        tp = ['add_numeric', 'multiply_numeric']
        cls = FeatureEng('D:/Higgs/unit_test/file_1.csv', 'infer', cat_col_name,
        'target', ap, tp, 2, 0.8)
        cls_df = cls.show()
        cls_result = cls.separate_features_from_label()

        df = pd.read_csv('D:/Higgs/unit_test/file_1.csv', header = 'infer')
        label_col_name = 'target'
        label_df = df.loc[:, [label_col_name]]
        features_df = df.drop(label_col_name, axis=1)
        df_result = (label_df, label_col_name, features_df)

        assert cls_df.shape == df.shape
        assert list(cls_df.dtypes) == list(df.dtypes)
        assert cls_df.ndim == df.ndim
        assert list(cls_df.columns) == list(df.columns)
        assert cls_df.size == df.size
        assert cls_df.equals(df)

        assert cls_result[1] == df_result[1], 'label_col_name'
        assert cls_result[0].shape == df_result[0].shape, 'label_df_shape'
        assert cls_result[2].shape == df_result[2].shape, 'features_df_shape'
        assert list(cls_result[0].dtypes) == list(df_result[0].dtypes), 'label_df_dtype'
        assert list(cls_result[2].dtypes) == list(df_result[2].dtypes), 'features_df_dtype'
        assert cls_result[0].equals(df_result[0])
        assert cls_result[2].equals(df_result[2])

    def test_numeric_features(self):
        '''Tests the numeric_features method in FeatureEng class'''
        cat_col_name = []
        ap = []
        tp = ['add_numeric', 'multiply_numeric']
        cls = FeatureEng('D:/Higgs/unit_test/file_1.csv', 'infer', cat_col_name,
        'target', ap, tp, 2, 0.8)
        cls_result = cls.numeric_features()

        df = pd.read_csv('D:/Higgs/unit_test/file_1.csv', header = 'infer')
        label_col_name = 'target'
        label_df = df.loc[:, [label_col_name]]
        features_df = df.drop(label_col_name, axis=1)
        if cat_col_name is None:
            numeric_df = features_df.copy()
        else:
            feat_df = df.drop(label_col_name, axis=1)
            cat_features = feat_df.loc[:, cat_col_name]
            numeric_df = feat_df.drop(cat_col_name, axis=1)

        assert cls_result.shape == numeric_df.shape
        assert list(cls_result.dtypes) == list(numeric_df.dtypes)
        assert cls_result.equals(numeric_df)

    def test_new_features(self):
        '''Tests the new_features method in FeatureEng class'''
        cat_col_name = []
        ap = []
        tp = ['add_numeric', 'multiply_numeric']
        cls = FeatureEng('D:/Higgs/unit_test/file_1.csv', 'infer', cat_col_name,
        'target', ap, tp, 2, 0.8)
        cls_result = cls.new_features()

        df = pd.read_csv('D:/Higgs/unit_test/file_1.csv', header = 'infer')
        label_col_name = 'target'
        label_df = df.loc[:, [label_col_name]]
        features_df = df.drop(label_col_name, axis=1)
        if cat_col_name is None:
            numeric_df = features_df.copy()
        else:
            feat_df = df.drop(label_col_name, axis=1)
            cat_features = feat_df.loc[:, cat_col_name]
            numeric_df = feat_df.drop(cat_col_name, axis=1)

        if numeric_df.shape[1] == numeric_df.select_dtypes(include=np.number).shape[1]:
            es = ft.EntitySet(id='id_1')
            es.entity_from_dataframe(entity_id='id_2', dataframe=numeric_df,
                                     make_index=True, index='new_index')
            feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='id_2',

                                                            agg_primitives= ap,
                                                            trans_primitives= tp,
                                                            max_depth= 2)
            if cat_col_name is not None:
                for col in cat_col_name:
                    feature_matrix[col] = cat_features[col].values
        else:
            raise ValueError("Data Frame contains non-numeric values")

        assert cls_result.shape == feature_matrix.shape
        assert list(cls_result.dtypes) == list(feature_matrix.dtypes)
        assert cls_result.equals(feature_matrix)

    def test_removing_correlated_features(self):
        '''Tests the remove_correlated_features method in FeatureEng class'''
        cat_col_name = []
        ap = []
        tp = ['add_numeric', 'multiply_numeric']
        cls = FeatureEng('D:/Higgs/unit_test/file_1.csv', 'infer', cat_col_name,
        'target', ap, tp, 2, 0.8)
        cls_result = cls.remove_correlated_features()

        df = pd.read_csv('D:/Higgs/unit_test/file_1.csv', header = 'infer')
        label_col_name = 'target'
        label_df = df.loc[:, [label_col_name]]
        features_df = df.drop(label_col_name, axis=1)
        if cat_col_name is None:
            numeric_df = features_df.copy()
        else:
            feat_df = df.drop(label_col_name, axis=1)
            cat_features = feat_df.loc[:, cat_col_name]
            numeric_df = feat_df.drop(cat_col_name, axis=1)

        if numeric_df.shape[1] == numeric_df.select_dtypes(include=np.number).shape[1]:
            es = ft.EntitySet(id='id_1')
            es.entity_from_dataframe(entity_id='id_2', dataframe=numeric_df,
                                     make_index=True, index='new_index')
            feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='id_2',
                                                            agg_primitives= ap,
                                                            trans_primitives= tp,
                                                            max_depth= 2)
            if cat_col_name is not None:
                for col in cat_col_name:
                    feature_matrix[col] = cat_features[col].values
        else:
            raise ValueError("Data Frame contains non-numeric values")

        col_corr = set()
        corr_matrix = feature_matrix.corr()
        for i in range(len(corr_matrix.columns)):
             for j in range(i):
                if ((abs(corr_matrix.iloc[i, j]) >= 0.8) and
                (corr_matrix.columns[j] not in col_corr)):
                    col_name = corr_matrix.columns[i]
                    col_corr.add(col_name)
                    if col_name in feature_matrix.columns:
                        del feature_matrix[col_name]

        assert cls_result.shape == feature_matrix.shape
        assert list(cls_result.dtypes) == list(feature_matrix.dtypes)
        assert cls_result.equals(feature_matrix)

    def test_df_with_new_features(self):
        '''Tests the df_with_new_features method in FeatureEng class'''
        cat_col_name = []
        ap = []
        tp = ['add_numeric', 'multiply_numeric']
        cls = FeatureEng('D:/Higgs/unit_test/file_1.csv', 'infer', cat_col_name,
        'target', ap, tp, 2, 0.8)
        cls_result = cls.df_with_new_features()

        df = pd.read_csv('D:/Higgs/unit_test/file_1.csv', header = 'infer')
        label_col_name = 'target'
        label_df = df.loc[:, [label_col_name]]
        features_df = df.drop(label_col_name, axis=1)
        if cat_col_name is None:
            numeric_df = features_df.copy()
        else:
            feat_df = df.drop(label_col_name, axis=1)
            cat_features = feat_df.loc[:, cat_col_name]
            numeric_df = feat_df.drop(cat_col_name, axis=1)

        if numeric_df.shape[1] == numeric_df.select_dtypes(include=np.number).shape[1]:
            es = ft.EntitySet(id='id_1')
            es.entity_from_dataframe(entity_id='id_2', dataframe=numeric_df,
                                     make_index=True, index='new_index')

            feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='id_2',
                                                            agg_primitives= ap,
                                                            trans_primitives= tp,
                                                            max_depth= 2)
            if cat_col_name is not None:
                for col in cat_col_name:
                    feature_matrix[col] = cat_features[col].values
        else:
            raise ValueError("Data Frame contains non-numeric values")

        col_corr = set()
        corr_matrix = feature_matrix.corr()
        for i in range(len(corr_matrix.columns)):
             for j in range(i):
                if ((abs(corr_matrix.iloc[i, j]) >= 0.8) and
                (corr_matrix.columns[j] not in col_corr)):
                    col_name = corr_matrix.columns[i]
                    col_corr.add(col_name)
                    if col_name in feature_matrix.columns:
                        del feature_matrix[col_name]

        if "target" in features_df.columns:
            raise ValueError(
                "Data Frame already contains target column")
        else:
            df_new_features = feature_matrix.copy()
            df_new_features['target'] = label_df[label_col_name].values

        assert cls_result.shape == df_new_features.shape
        assert list(cls_result.dtypes) == list(df_new_features.dtypes)
        assert cls_result.equals(df_new_features)

    def selector_save(self):
        '''Tests all the methods of FEatureSelect class'''
        cat_col_name = []
        ap = []
        tp = ['add_numeric', 'multiply_numeric']
        num_features = 6
        pcls = FeatureEng('D:/Higgs/unit_test/file_1.csv', 'infer', cat_col_name,
        'target', ap, tp, 2, 0.8)
        pcls.new_features()
        cls = FeatureSelect(pcls.feature_matrix, num_features)
        cor_cls_result = cls.cor_pearson_selector()
        chi_cls_result = cls.chi_square_selector()
        log_cls_result = cls.log_reg_selector()
        rf_cls_result = cls.random_forest_selector()
        lgb_cls_result = cls.LGBM_selector()
        et_cls_result = cls.Extra_Trees_selector()
        comb_cls_result = cls.combine_selector()
        save_cls_result = cls.save_df_selected_columns()

        df = pd.read_csv('D:/Higgs/unit_test/file_1.csv', header = 'infer')
        label_col_name = 'target'
        label_df = df.loc[:, [label_col_name]]
        features_df = df.drop(label_col_name, axis=1)
        if cat_col_name is None:
            numeric_df = features_df.copy()
        else:
            feat_df = df.drop(label_col_name, axis=1)
            cat_features = feat_df.loc[:, cat_col_name]
            numeric_df = feat_df.drop(cat_col_name, axis=1)

        if numeric_df.shape[1] == numeric_df.select_dtypes(include=np.number).shape[1]:
            es = ft.EntitySet(id='id_1')
            es.entity_from_dataframe(entity_id='id_2', dataframe=numeric_df,
                                     make_index=True, index='new_index')

            feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='id_2',

                                                            agg_primitives= ap,
                                                            trans_primitives= tp,
                                                            max_depth= 2)
            if cat_col_name is not None:
                for col in cat_col_name:
                    feature_matrix[col] = cat_features[col].values
        else:
            raise ValueError("Data Frame contains non-numeric values")

        X = feature_matrix.iloc[:, :-1]
        y = feature_matrix.iloc[:, -1]
        X_norm = MinMaxScaler().fit_transform(X)
        feature_name = X.columns.tolist()
        num_feats = num_features
        cor_list = []

        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)

        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_feature = X.iloc[:, np.argsort(
            np.abs(cor_list))[-num_feats:]].columns.tolist()

        cor_support = [
            True if i in cor_feature else False for i in feature_name]

        chi_selector = SelectKBest(chi2, k=num_features)
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        chi_feature = X.loc[:, chi_support].columns.tolist()

        embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"),
                                                   max_features=num_features)
        embeded_lr_selector.fit(X_norm, y)
        embeded_lr_support = embeded_lr_selector.get_support()
        embeded_lr_feature = X.loc[:,
                            embeded_lr_support].columns.tolist()

        embeded_rf_selector = SelectFromModel(RandomForestClassifier
		(n_estimators=100, random_state=1), max_features=num_features)
        embeded_rf_selector.fit(X, y)
        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()

        lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32,
        colsample_bytree=0.2, reg_alpha=3, reg_lambda=1,
        min_split_gain=0.01, min_child_weight=40, random_state=1)

        embeded_lgb_selector = SelectFromModel(
        lgbc, max_features = num_features)
        embeded_lgb_selector.fit(X, y)
        embeded_lgb_support = embeded_lgb_selector.get_support()
        embeded_lgb_feature = X.loc[:,
                                embeded_lgb_support].columns.tolist()

        extra_trees = ExtraTreesClassifier(random_state=1)
        extra_trees_selector = SelectFromModel(
            extra_trees, max_features=num_features)
        extra_trees_selector.fit(X, y)
        extra_trees_support = extra_trees_selector.get_support()
        extra_trees_feature = X.loc[:,
                                extra_trees_support].columns.tolist()

        feature_selection_df = pd.DataFrame({'Feature': feature_name,
        'Pearson': cor_support, 'Chi-2': chi_support,
        'Logistics': embeded_lr_support,
		'Random Forest': embeded_rf_support,
        'LightGBM': embeded_lgb_support,
		'Extra_trees': extra_trees_support})

        feature_selection_df['Total'] = np.sum(
            feature_selection_df, axis=1)
        feature_selection_df = feature_selection_df.sort_values(
            ['Total', 'Feature'], ascending=False)
        feature_selection_df.index = range(
            1, len(feature_selection_df)+1)

        col_num = []
        for i in range(len(feature_matrix.columns)):
            while i < len(feature_matrix.columns)-1:
                if (feature_matrix.columns[i] not in
                    feature_selection_df.head(num_feats).loc[:, 'Feature'].values):
                    col_num.append(i)
                else:
                    pass
                i += 1

        df_selected_columns = feature_matrix.drop(
            feature_matrix.columns[col_num], axis=1, inplace=False)
        df_selected_columns.to_csv('data_features_final.csv', index=False)

        assert len(cor_cls_result[0]) == len(cor_support)
        assert list(cor_cls_result[0].dtypes) == list(cor_support.dtypes)
        assert cor_cls_result[0] == cor_support
        assert len(cor_cls_result[1]) == len(cor_feature)
        assert list(cor_cls_result[1].dtypes) == list(cor_feature.dtypes)
        assert cor_cls_result[1] == cor_feature

        assert len(chi_cls_result) == len(chi_feature)
        assert chi_cls_result == chi_feature

        assert len(log_cls_result) == len(embeded_lr_feature)
        assert log_cls_result == embeded_lr_feature

        assert len(rf_cls_result) == len(embeded_rf_feature)
        assert rf_cls_result == embeded_rf_feature

        assert len(lgb_cls_result) == len(embeded_lgb_feature)
        assert lgb_cls_result == embeded_lgb_feature

        assert len(et_cls_result) == len(extra_trees_feature)
        assert et_cls_result == extra_trees_feature

        assert comb_cls_result.equals(feature_selection_df.head(num_features))
        assert save_cls_result.shape == df_selected_columns.shape
        assert list(save_cls_result.dtypes) == list(df_selected_columns.dtypes)
        assert save_cls_result.equals(df_selected_columns)