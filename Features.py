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


class FeatureEng():
    """
    creates new feature columns for dataset based on selected aggregations and transformations
    """
    def __init__(self, filepath, header_type, categorical_col_name=None):
        
        self.file_path = filepath
        self.df = pd.read_csv(filepath, header=header_type)
        self.df.columns = map(str, self.df.columns)  #covert title of columns to string type
        self.categorical_col_name = categorical_col_name
    
    #show complete dataframe
    def show(self):
        """
        shows the imported dataframe
        """
        
        return self.df
    
    def separate_features_from_label(self, label_col_name):
        """
        separates features from label column
        
        Parameters
        ----------
        label_col_name (str) : the name of label(output) column of dataframe
        
        Returns
        ----------
        label_df : dataframe containing only label(output) column
        label_col_name : column name of label column
        features : dataframe containing only features columns

        """
        self.label_df = self.df.loc[:,[label_col_name]]
        self.label_col_name = label_col_name
        self.features_df = self.df.drop(label_col_name,axis=1)
        
        return self.label_df, self.label_col_name, self.features_df


    
    def numeric_features(self):
        """
        separates numeric features, from categorical features and label columns, 
        and saves them into 3 separate dataframes 
        
        Parameters
        ----------
        categorical_col_name (str) : the name of categorical columns
        
        Returns
        ----------
        cat_features : dataframe containing only categorical features columns
        numeric_features : dataframe containing only numerical features columns
        categorical_col_name : column name of categorical columns
        

        """
        if self.categorical_col_name == None:
            self.numeric_features = self.features_df.copy()
            return self.numeric_features
            
        else:
            self.feat_df = self.df.drop(self.label_col_name,axis=1)
        #self.categorical_col_name = categorical_col_name
        #self.categorical_col_name = optional
            self.cat_features = self.feat_df.loc[:,[self.categorical_col_name]]
        #self.cat_features = self.feat_df.loc[:,[optional]]
            self.numeric_features = self.feat_df.drop(self.categorical_col_name,axis=1)
        #self.numeric_features = self.feat_df.drop(optional,axis=1)

            return self.cat_features, self.numeric_features, self.categorical_col_name 
        
    
    
    # Implement Feature tools to create new features
    def new_features(self, list_agg_primitives=None, list_trans_primitives=['multiply_numeric'],
    max_depth_value=1):
        
        """
        creates new features using current numeric features. 
        It only accepts numeric features as input.
        
        Parameters
        ----------
        list_agg_primitives(list[str or AggregationPrimitive], optional): 
            list of Aggregation Feature types to apply.
                example types of aggreation for ft.dfs: 
                ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]

        list_trans_primitives (list[str or TransformPrimitive], optional): 
            List of Transform Feature functions to apply.
                example types of aggreation for ft.dfs: 
                ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]  
        For more available perimitives : https://primitives.featurelabs.com/
        
        max_depth_value (int) : Maximum allowed depth of features.
        
        Returns
        ----------
        self.feature_matrix : dataframe containing all the old features and new synthetized features

        """
    
        if self.numeric_features.shape[1] == self.numeric_features.select_dtypes(include=np.number).shape[1]:
            # Make an entityset and add the entity
            es = ft.EntitySet(id = 'id_1')
            es.entity_from_dataframe(entity_id='id_2', dataframe=self.numeric_features , 
                         make_index=True, index='new_index')
            # Run deep feature synthesis
            self.feature_matrix, self.feature_defs = ft.dfs(entityset=es, target_entity='id_2',

                                            agg_primitives=list_agg_primitives,
                                            trans_primitives=list_trans_primitives,
                                            max_depth=max_depth_value)
            
            #Add categorical features back to the features dataframe
            if self.categorical_col_name != None:
                for col in self.categorical_col_name:
                    self.feature_matrix[col] = self.cat_features[col].values

            return self.feature_matrix

        else:

            raise ValueError( "Data Frame contains non-numeric values")


    def remove_correlated_features(self, threshold):
        """
        removes highly correlated features

        Parameters
        ----------
        threshold (int) : number between 0 and 1 which determines maximum features correlation limit 
        
        Returns
        ----------
        feature_matrix : dataframe containing all the features that have correlation 
            less than threshold value

        """

        if (threshold>0) or (threshold<1):
            self.col_corr = set() # Set of all the names of deleted columns
            self.corr_matrix = self.feature_matrix.corr()
            for i in range(len(self.corr_matrix.columns)):
                for j in range(i):
                    if (abs(self.corr_matrix.iloc[i, j]) >= threshold) and (self.corr_matrix.columns[j] not in self.col_corr):
                        self.col_name = self.corr_matrix.columns[i] # getting the name of column
                        self.col_corr.add(self.col_name)
                        if self.col_name in self.feature_matrix.columns:
                            del self.feature_matrix[self.col_name] # deleting the column from the dataset

            return self.feature_matrix

        else:
            raise ValueError( "remove_correlated_features function accepts integers in range of (0,1)")  

    
    def df_with_new_features(self):
        """
        adds the label(output) column to the dataframe with new features
        
        Returns
        ----------
        df_new_features : dataframe containing the features and label

        """
        if self.label_col_name in self.feature_matrix.columns:
            raise ValueError( "Data Frame already contains {} column".format(self.label_col_name))
            
        else:
            self.df_new_features = self.feature_matrix.copy()
            self.df_new_features['target'] = self.label_df[self.label_col_name].values
        return self.df_new_features


    
class FeatureSelect(FeatureEng):
    """
    combines several methods to choose the best features

    Parameters
    ----------
    df_new_features(dataframe) : processed dataframe from FeatureEng Parent class
    num_feats(int) : number of features to be selected
    
    Returns
    ----------
    X : features column 
    y : label column
    X_norm : normalized data using MinMaxScaler method
    feature_name : list containing name of features
  
    """
    #preparing data for model
    def __init__(self, df_new_features, num_features):

        FeatureEng.df_new_features = df_new_features
        self.X = self.df_new_features.iloc[:,:-1]  #independent columns
        self.y = self.df_new_features.iloc[:,-1]   #target column
        self.X_norm = MinMaxScaler().fit_transform(self.X)
        self.feature_name = self.X.columns.tolist()
        self.num_feats = num_features
        

    
    def cor_pearson_selector(self):
        """
        selects the top n=num_feats features using Pearson’s correlation between featurs and the target. 
        
        Returns
        ----------
        cor_support :
        cor_feature :

        """
        self.cor_list = []
    

    # calculate the correlation with y for each feature

        for i in self.X.columns.tolist():

            cor = np.corrcoef(self.X[i], self.y)[0, 1]

            self.cor_list.append(cor)

    # replace NaN with 0

        self.cor_list = [0 if np.isnan(i) else i for i in self.cor_list]

    # feature name

        self.cor_feature = self.X.iloc[:,np.argsort(np.abs(self.cor_list))[-self.num_feats:]].columns.tolist()

    # feature selection? 0 for not select, 1 for select

        self.cor_support = [True if i in self.cor_feature else False for i in self.feature_name]

        return self.cor_support, self.cor_feature


    
    def chi_square_selector(self):
        """
        selects top n=num_feats features using Chi-Squared method.

        Returns
        ----------
        chi_feature :

        """

        self.chi_selector = SelectKBest(chi2, k=self.num_feats)

        self.chi_selector.fit(self.X_norm, self.y)

        self.chi_support = self.chi_selector.get_support()

        self.chi_feature = self.X.loc[:,self.chi_support].columns.tolist()

        return self.chi_feature



    #feature selection using Recursive Feature Elimination
    def recursive_selector(self):
        """
        selects top n=num_feats features using Recursive Feature Elimination method.

        Returns
        ----------
        rfe_feature :

        """

        self.rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=self.num_feats, step=10, verbose=5)

        self.rfe_selector.fit(self.X_norm, self.y)

        self.rfe_support = self.rfe_selector.get_support()

        self.rfe_feature = self.X.loc[:,self.rfe_support].columns.tolist()

        return self.rfe_feature


    def log_reg_selector(self):
        """
        selects top n=num_feats features using logistic regression method.

        Returns
        ----------
        embeded_lr_feature :

        """
       
        self.embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), 
                                                   max_features=self.num_feats)

        self.embeded_lr_selector.fit(self.X_norm, self.y)

        self.embeded_lr_support = self.embeded_lr_selector.get_support()

        self.embeded_lr_feature = self.X.loc[:,self.embeded_lr_support].columns.tolist()

        return self.embeded_lr_feature


    def random_forest_selector(self):
        """
        selects top n=num_feats features using Random Forest method.

        Returns
        ----------
        embeded_rf_feature :

        """

        self.embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=1), 
                                                   max_features=self.num_feats)

        self.embeded_rf_selector.fit(self.X, self.y)

        self.embeded_rf_support = self.embeded_rf_selector.get_support()

        self.embeded_rf_feature = self.X.loc[:,self.embeded_rf_support].columns.tolist()

        return self.embeded_rf_feature


    #feature selection using Light GBM
    def LGBM_selector(self):
        """
        selects top n=num_feats features using Light GBM method.

        Returns
        ----------
        embeded_lgb_feature :

        """
        self.lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, 
                                 colsample_bytree=0.2, reg_alpha=3, reg_lambda=1,
                                 min_split_gain=0.01, min_child_weight=40, random_state=1)

        self.embeded_lgb_selector = SelectFromModel(self.lgbc, max_features=self.num_feats)

        self.embeded_lgb_selector.fit(self.X, self.y)

        self.embeded_lgb_support = self.embeded_lgb_selector.get_support()

        self.embeded_lgb_feature = self.X.loc[:,self.embeded_lgb_support].columns.tolist()

        return self.embeded_lgb_feature
    
    
    def Extra_Trees_selector(self):
        """
        selects top n=num_feats features using Extra Tree Classifier.

        Returns
        ----------
        extra_trees_feature :

        """
        
        self.extra_trees = ExtraTreesClassifier(random_state=1)
        
        self.extra_trees_selector = SelectFromModel(self.extra_trees, max_features=self.num_feats)
        
        self.extra_trees_selector.fit(self.X, self.y)
        
        self.extra_trees_support = self.extra_trees_selector.get_support()
        
        self.extra_trees_feature = self.X.loc[:,self.extra_trees_support].columns.tolist()

        return self.extra_trees_feature


    # put all selection together
    def combine_selector(self):
        """
        selects top n=num_feats features using combinations of methods.

        Returns
        ----------
        feature_selection_df.head(num_feats) : dataframe containing the top features with their 
            number of votes for each feature selection method.

        """


        self.feature_selection_df = pd.DataFrame({'Feature':self.feature_name, 'Pearson':self.cor_support, 
                                                  'Chi-2':self.chi_support, 'RFE':self.rfe_support, 
                                                  'Logistics':self.embeded_lr_support, 'Random Forest':self.embeded_rf_support, 
                                                  'LightGBM':self.embeded_lgb_support,'Extra_trees':self.extra_trees_support})
                                                  
        # count the selected times for each feature

        self.feature_selection_df['Total'] = np.sum(self.feature_selection_df, axis=1)

        # display the top features

        self.feature_selection_df = self.feature_selection_df.sort_values(['Total','Feature'] , ascending=False)

        self.feature_selection_df.index = range(1, len(self.feature_selection_df)+1)

        return self.feature_selection_df.head(self.num_feats)

       
    def save_df_selected_columns(self):
        """
        saves final dataset with top n=num_feats selected features.

        Returns
        ----------
        df_selected_columns : dataframe containing the top features and label to be used for next steps.

        """
        
        self.col_num = []

        for i in range(len(self.df_new_features.columns)):
            while i < len(self.df_new_features.columns)-1: #excluding target column
                if self.df_new_features.columns[i] not in (self.feature_selection_df.head(self.num_feats).loc[:,'Feature'].values):
                    self.col_num.append(i)
                    
                else:
                    pass
                
                i+=1
        
        self.df_selected_columns = self.df_new_features.drop(self.df_new_features.columns[self.col_num], axis=1, inplace=False)
        self.df_selected_columns.to_csv('data_features_final.csv',index=False)
            
        return self.df_selected_columns
