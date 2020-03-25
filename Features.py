import random
import pandas as pd
import numpy as np
import featuretools as ft
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier



class FeatureEng():
    """creates new feature columns for dataset based on selected aggregations and transformations
    """
    def __init__(self, filepath, header_type):
        
        self.file_path = filepath
        self.df = pd.read_csv(filepath, header=header_type)
        #covert title of columns to string type
        self.df.columns = map(str, self.df.columns)

    #show complete dataframe
    def show(self):
        return self.df

    #create a data frame of only features (drop target column)
    def separate_features_from_label(self, target_col_name):
        
        self.label_df = self.df.loc[:,[target_col_name]]
        self.features_df = self.df.drop(target_col_name,axis=1)
        return self.label_df, self.features_df

    # Implement Feature tools to create new features
    def new_features(self, list_agg_primitives, list_trans_primitives):
        if self.features_df.shape[1] == self.features_df.select_dtypes(include=np.number).shape[1]:
            # Make an entityset and add the entity
            es = ft.EntitySet(id = 'id_1')
            es.entity_from_dataframe(entity_id='id_2', dataframe=self.features_df , 
                         make_index=True, index='new_index')
            # Run deep feature synthesis
            self.feature_matrix, self.feature_defs = ft.dfs(entityset=es, target_entity='id_2',

                                            agg_primitives=list_agg_primitives,
                                            trans_primitives=list_trans_primitives)

            return self.feature_matrix, self.feature_defs

        else:
            raise ValueError( "Data Frame contains non-numeric values")


    #adding the target(output) column to the dataframe with new features
    def df_with_new_features(self, target_col_name):
        if target_col_name in self.feature_matrix.columns:
            raise ValueError( "Data Frame already contains {} column".format(target_col_name))
            
        else:
            self.feature_matrix['target'] = self.label_df[target_col_name].values
            return self.feature_matrix

    
class FeatureSelect(FeatureEng):
    """combines several methods to choose the best features
    """
#preparing data for model
def __init__(self, split_ratio):
    self.X = FeatureEng.feauture_matrix.iloc[:,:-1]  #independent columns
    self.y = FeatureEng.feauture_matrix.iloc[:,-1]   #target column
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = split_ratio, random_state = 1)
    self.X_norm = MinMaxScaler().fit_transform(self.X_train)
    self.feature_name = self.X_train.columns.tolist()

#feature selection using Pearson’s correlation
#We check the absolute value of the Pearson’s correlation between the target 
#and numerical features in our dataset. We keep the top n features based on this criterion.
def cor_pearson_selector(self,num_feats):

    self.cor_list = []
    

    # calculate the correlation with y for each feature

    for i in self.X_train.columns.tolist():

        cor = np.corrcoef(self.X_train[i], self.y_train)[0, 1]

        self.cor_list.append(cor)

    # replace NaN with 0

    self.cor_list = [0 if np.isnan(i) else i for i in self.cor_list]

    # feature name

    self.cor_feature = self.X_train.iloc[:,np.argsort(np.abs(self.cor_list))[-num_feats:]].columns.tolist()

    # feature selection? 0 for not select, 1 for select

    self.cor_support = [True if i in self.cor_feature else False for i in self.feature_name]

    return self.cor_support, self.cor_feature


#feature selection using Chi-Squared method
def chi_square_selector(self,num_feats):

    self.chi_selector = SelectKBest(chi2, k=num_feats)

    self.chi_selector.fit(self.X_norm, self.y_train)

    self.chi_support = self.chi_selector.get_support()

    self.chi_feature = self.X_train.loc[:,self.chi_support].columns.tolist()

    return self.chi_feature



#feature selection using Recursive Feature Elimination
def recursive_selector(self,num_feats):

    self.rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)

    self.rfe_selector.fit(self.X_norm, self.y_train)

    self.rfe_support = self.rfe_selector.get_support()

    self.rfe_feature = self.X_train.loc[:,self.rfe_support].columns.tolist()

    return self.rfe_feature


#feature selction using logistic regression
def log_reg_selector(self,num_feats):

    self.embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)

    self.embeded_lr_selector.fit(self.X_norm, self.y_train)

    self.embeded_lr_support = self.embeded_lr_selector.get_support()

    self.embeded_lr_feature = self.X_train.loc[:,self.embeded_lr_support].columns.tolist()

    return self.embeded_lr_feature



# feature selection using Random Forest
def random_forest_selector(self,num_feats):

    self.embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)

    self.embeded_rf_selector.fit(self.X_train, self.y_train)

    self.embeded_rf_support = self.embeded_rf_selector.get_support()

    self.embeded_rf_feature = self.X_train.loc[:,self.embeded_rf_support].columns.tolist()

    return self.embeded_rf_feature


#feature selection using Light GBM
def LGBM_selector(self,num_feats):
    self.lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,

            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)


    self.embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)

    self.embeded_lgb_selector.fit(self.X_train, self.y_train)

    self.embeded_lgb_support = self.embeded_lgb_selector.get_support()

    self.embeded_lgb_feature = self.X_train.loc[:,self.embeded_lgb_support].columns.tolist()

    return self.embeded_lgb_feature



# put all selection together
def combine_selector(self,num_feats):
    self.feature_selection_df = pd.DataFrame({'Feature':self.feature_name, 'Pearson':self.cor_support, 'Chi-2':self.chi_support, 'RFE':self.rfe_support, 'Logistics':self.embeded_lr_support,

                                    'Random Forest':self.embeded_rf_support, 'LightGBM':self.embeded_lgb_support})

    # count the selected times for each feature

    self.feature_selection_df['Total'] = np.sum(self.feature_selection_df, axis=1)

    # display the top features

    self.feature_selection_df = self.feature_selection_df.sort_values(['Total','Feature'] , ascending=False)

    self.feature_selection_df.index = range(1, len(self.feature_selection_df)+1)

    return feature_selection_df.head(num_feats)
