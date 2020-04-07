import numpy as np
import pandas as pd
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


class Preprocessor():
    
    """
    A class that compiles several functions that preprocess both numeric and categorical data for a general dataset. 
    
    Includes functions that handle duplicates, impute/drop missing values and scale/normalize the data. 
    
    It relies on 'dask' library for parallel computing and as a result, can handle large datasets.
    """
    
    def __init__(self, numeric_features, target_feature, categorical_features = None):
        
        """
        Initializes an instance of the Preprocessor class.
        
        Parameters
        ----------

        numeric_features : list of strings
            A list containing the column labels corresponding to numeric features,
        categorical features: list of strings, default = None
            A list containing the column labels corresponding to categorical features
        target_feature: string
            Column label corresponding to the target feature
        """
        
        self.target_feature = target_feature
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        
        print('Preprocessor ready...')
        
    def remove_duplicates(self, dataframe):
        
        """
        Method that finds AND removes duplicated observations in a dataframe
        
        Parameters
        ----------
        dataframe : Dask dataframe
            A dataframe for which duplicates must be removed

        Returns
        ----------
        df : Dask dataframe
            A dataframe with duplicates, if any, are removed.
        
        """
        
        df = dataframe.copy()
        df = df.drop_duplicates()
        return df
    
    def target_encoder(self, dataframe):
    
        """
        Method that handles the preprocessing of the target feature
        
        Parameters
        ----------
        dataframe : Dask dataframe
            A dataframe for which the target feature must be encoded

        Returns
        ----------
        df : Dask dataframe
            A dataframe with the target feature encoded
        
        """
        df = dataframe.copy()
        
        le = LabelEncoder()
        df[self.target_feature] = le.fit_transform(df[self.target_feature])
        
        return df
    
    def missing_values(self, dataframe, drop_threshold = 50, category_threshold = 1, method = 'fill'):
        
        """
        Method that handles missing data in numeric and categorical features
        
        Parameters
        ----------
        dataframe : Dask dataframe
            A dataframe for which the target feature must be encoded
        drop_threshold : float
            Percentage of missing values in a feature below which a column will be dropped
        category_threshold: integer
            Count of categories below which (or equal) a category will be grouped together as 'Other' for a categorical feature
        method: string
            How to handle missing values in a feature, options are 'fill' and 'drop'
                fill: the missing values are imputed with the mean for the numeric features and the mode for categorical features
                drop: the missing values are dropped entirely for both numeric and categorical features

        Returns
        ----------
        df : Dask dataframe
            A dataframe with the missing values imputed or dropped
    
        """
        
        if method not in ['fill', 'drop']:
            raise ValueError("method must be one of: fill, drop")
        
        try:
            int(drop_threshold)
        except:
            raise TypeError("Threshold value must be numeric")
            
        if isinstance(category_threshold, int):
            pass
        else:
            raise TypeError('Threshold value must be an integer')
            
        
        df = dataframe.copy()
        missing = df.isnull().sum()
        if missing.any().compute():
            print("Missing values detected")
            if method == 'drop':
                df = df.dropna()
                
            elif method == 'fill':
                if drop_threshold!=None:
                    percent_missing = ((missing/df.index.size)*100).compute()
                    drop_list = list(percent_missing[percent_missing > drop_threshold].index)
                    df = df.drop(drop_list, axis = 1)
   
        #NUMERIC COLUMNS
        
                if df[self.numeric_features].isnull().sum().any().compute():
                    means = df[self.numeric_features].mean().compute()
                    df[self.numeric_features] = df[self.numeric_features].fillna(means)

            
        #CATEGORICAL COLUMNS
                if df[self.categorical_features].isna().sum().any().compute():
                    for i in list(self.categorical_features):
                        category_count = df[i].value_counts().compute()
                        mode = category_count.index[0]
                        df[i] = df[i].fillna(mode)
                        distinct = list(category_count[category_count <= category_threshold].index)
                        df[i] = df[i].replace(distinct, 'Other')
                        
                        
        return df
        
    def scale(self, dataframe, method = 'standardscaler', feature_range = (0, 1), train = True):
        
        """
        Method that handles missing data in numeric and categorical features
        
        Parameters
        ----------
        dataframe : Dask dataframe
            A dataframe for which the target feature must be encoded
        method: string, default = 'standardscaler'
            Which scaling algorithm to use on the data
                standardscaler: applies Scikit-Learn's StandardScaler; data is centered by subtracting the mean and scaled by dividing by 
                                the standard deviation
                minmax: applies Scikit-Learn's MinMaxScaler; data is scaled to a range specified by the feature_range parameter.
                        Calculation: [(Observation - Observation(minimum)/ (Observation(maximum) - Observation (minimum)))]*(upper_limit - 
                        lower_limit) + lower_limit
                robust: applies Scikit-Learn's RobustScaler; data is centered by subtracting the median and scaled by dividing by the
                        interquartile range. Used for data with outliers.
        feature_range: tuple(lower_limit, upper_limit)
            Specifies the upper and lower limits of scaling; applicable only for the minmax method only.
        train: bool, default = True
            Specifies if the scaler is handling training or testing data
            
        Returns
        ----------
        df : Dask dataframe
            A dataframe that has been scaled
    
        """
        
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Lower limit must be less than upper limit for scaling.") 
        if method not in ['standardscaler', 'minmax', 'robust']:
            raise ValueError("method must be one of: standardscaler, minmax, or robust")
        
        df = dataframe.copy()
        
        if train:
            if method == 'standardscaler':
                scaler = dask_ml.preprocessing.StandardScaler()
            elif method == 'minmax':
                scaler = dask_ml.preprocessing.MinMaxScaler(feature_range = feature_range)
            elif method == 'robust':
                scaler = dask_ml.preprocessing.RobustScaler()
            self.scaler = scaler
            df[self.numeric_features] = scaler.fit_transform(df[self.numeric_features])

        else:
            df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])
        
        return df

    def transformations(self, dataframe, method = 'yeo-johnson', train = True):
        
        """
        Method that handles transforming data to more Gaussian-like distributions and stabilized variance using scikit-learn PowerTransformer.
        Please use the class's 'scale' method if you wish to apply scaling before using this transformer
        
        Parameters
        ----------
        dataframe : Dask dataframe
            A dataframe for which the target feature must be encoded
        method: string, default = 'yeo-johnson'
            Which transformation algorithm to use on the data
                yeo-johnson: works for both positive and negative values
                box-cox: works for positive values only 
        train: bool, default = True
            Specifies if the scaler is handling training or testing data
        
        Returns
        ----------
        df : Dask dataframe
            A dataframe for which numeric features have been transformed to Gaussian-like distributions and stabilized variance
    
        """

        if method not in ['yeo-johnson', 'box-cox']:
            raise ValueError("method must be one of: yeo-johnson, box-cox")

        df = dataframe.copy()
        
        if train:
            transformer = sklearn.preprocessing.PowerTransformer(method = method, standardize = False)
            df[self.numeric_features] = dd.from_array(df[self.numeric_features].map_partitions(transformer.fit_transform))
            self.transformer = transformer
        else:
            df[self.numeric_features] = dd.from_array(df[self.numeric_features].map_partitions(self.transformer.transform))

        return df

