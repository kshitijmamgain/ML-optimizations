import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dask
from dask import compute
import dask.dataframe as dd
from dask_ml.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from dask_ml.decomposition import PCA

import missingno

class Visualizer():
    
    """
    A class that compiles several functions that visualize both numeric and categorical data for a general dataset. 
    
    Includes functions that display missing values, correlations and results from principal component analysis 
    
    """
    
    def __init__(self, target_feature, numeric_features, categorical_features = None):

        """
        Initializes an instance of the Visualizer class.
        
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
        print("Data Visualizer ready...")
        
    def missing_values(self, dataframe, byclass = False):
        
        """
        Creates a bar plot for the count of missing values
        
        Parameters
        ----------
        dataframe : a Dask dataframe
            A Dask dataframe for which missing values are to be visualized
        byclass: bool, default = False
            Specifies whether separate plots should be made for each class of the target feature
        
        """        
        
        df = dataframe.copy()
        
        if byclass:
            classes = df[self.target_feature].unique()
            for c in classes:
                graph = dataframe[dataframe[self.target_feature] == c].compute()
                plt.figure()
                plt.title('Missing Values for Class - ' + str(c))
                plt.xlabel('Features')
                plt.ylabel('Missing Observations')
                missingno.bar(graph)
        else:
            plt.figure()
            plt.title('Missing Values in Dataset')
            plt.xlabel('Features')
            plt.ylabel('Missing Observations')
            missingno.bar(graph) 
        
    def correlations(self, dataframe, method = 'pearson'):
        
        """
        Creates a correlations heatmap for the features of the datset
        
        Parameters
        ----------
        dataframe: a Dask dataframe
            A Dask dataframe for which missing values are to be visualized
        method: string
            The method by which to calculate the correlation coefficients. Must be one of ['pearson', 'kendall', 'spearman']
        """

        if method not in ['pearson', 'kendall', 'spearman']:
            raise ValueError("method must be one of: pearson, kendall, spearman")
        
        df = dataframe.copy()
        corr = df.corr(method = method).compute()
        mask = np.tril(np.ones(corr.shape[1])).astype(np.bool)
        plt.figure(figsize = (20, 10))
        sns.heatmap(corr.where(mask), annot = True, fmt = '.2f')
        plt.title('Correlation Matrix')
        
    def principal_components(self, dataframe, plot_type = 'explained_variance', exclude_target = False):
        
        """
        Creates plots related to principal component analysis performed on the dataset
        
        Parameters
        ----------
        dataframe: a Dask dataframe
            A Dask dataframe for which missing values are to be visualized
        plot_type: string
            The type of principal component plot to produce. Must be one of ["explained variance", "scatter"]
        """

        if plot_type not in ['explained_variance', 'scatter']:
            raise ValueError("method must be one of: explained_variance, scatter")
        
        df = dataframe.copy()
        pc_features = list(df.columns).copy()
        
        target = df[self.target_feature].values.compute()
        distinct_target = len(np.bincount(target.astype(int)))
        
        if exclude_target:
            pc_features.remove(self.target_feature)
        pc = dask_ml.decomposition.PCA(n_components = len(pc_features), svd_solver = 'randomized')
        
        df = df[pc_features]
        
        projected = pc.fit_transform(df.to_dask_array(lengths = True))
        
        variance = np.cumsum(pc.explained_variance_ratio_*100)
        
        if plot_type == 'explained_variance':
            plt.bar(range(1, len(pc_features)+1), pc.explained_variance_ratio_*100, alpha = 0.5,
            align = 'center', label = 'Individual Explained Variance (%)')
            plt.step(range(1, len(pc_features)+1), variance, where = 'mid',
                     label = 'Cumulative Explained Variance (%)')
            plt.title('Variance Explained By Principal Components')
            plt.ylabel('Explained Variance Ratio (%)')
            plt.xlabel('Principal Component')
            plt.legend(loc = 'best')
            plt.show();

        elif plot_type == 'scatter':
            plt.scatter(projected[:, 0], projected[:, 1],
            c = target, edgecolor = 'none', alpha = 0.5,
            cmap = plt.cm.get_cmap('Spectral', distinct_target))
            plt.title('Scatter of Two Principal Components (By Class)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Princiapl Component 2')
            plt.colorbar()
            plt.show();        