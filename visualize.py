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
    
    Includes functions that plot missing values, distributions, correlations and principal component plots
    
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
        self.pc = None
        self.projected = None
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
        
    def distributions(self, dataframe, byclass = False, bins = 20):
        
        """
        Creates a plots that explore the distributions of both numerical and categorical features
        
        Parameters
        ----------
        dataframe: a Dask dataframe
            A Dask dataframe for which missing values are to be visualized
        byclass: bool, default = False
            Whether to plot distributions for each class separately
        bins: int, default = 20
            Numeber of bins to plot for the histograms of numeric features
        """
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError('Bins value must be an integer')
        
        df = dataframe.copy()
        df = df.compute()
        
        if byclass:
            col = self.target_feature
        else:
            col = None
            
        for i in dataframe.columns:
            if i in self.numeric_features:
                plot = sns.FacetGrid(data = df, col = col)
                plot.map(plt.hist, i, density = True, bins = bins)
            if i in self.categorical_features:
                plot = sns.FacetGrid(data = df, col = col)
                plt.map(sns.countplot, i)
        
        
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
        
    def principal_components(self, dataframe, exclude_target = True):
        
        """
        Creates plots related to principal component analysis performed on the dataset
        
        Parameters
        ----------
        dataframe: a Dask dataframe
            A Dask dataframe for which missing values are to be visualized
        exclude_target: bool, default = True
            Whether to exclude the target feature from the analysis
        """
        
        df = dataframe.copy()
        pc_features = list(df.columns).copy()
        
        target = df[self.target_feature].values.compute()
        distinct_target = len(np.bincount(target.astype(int)))
        
        if exclude_target:
            pc_features.remove(self.target_feature)
            
        df = df[pc_features]

        pc = dask_ml.decomposition.PCA(n_components = len(pc_features), svd_solver = 'randomized')
        projected = pc.fit_transform(df.to_dask_array(lengths = True))

        variance = np.cumsum(pc.explained_variance_ratio_*100)
        
        #EXPLAINED VARIANCE PLOT
        plt.figure()
        plt.bar(range(1, len(pc_features)+1), pc.explained_variance_ratio_*100, alpha = 0.5,
        align = 'center', label = 'Individual Explained Variance (%)')
        plt.step(range(1, len(pc_features)+1), variance, where = 'mid',
                 label = 'Cumulative Explained Variance (%)')
        plt.title('Variance Explained By Principal Components')
        plt.ylabel('Explained Variance Ratio (%)')
        plt.xlabel('Principal Component')
        plt.legend(loc = 'best')
        plt.show();

        #SCATTERPLOT
        plt.figure()
        plt.scatter(projected[:, 0], projected[:, 1],
        c = target, edgecolor = 'none', alpha = 0.5,
        cmap = plt.cm.get_cmap('Spectral', distinct_target))
        plt.title('Scatter of Two Principal Components (By Class)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Princiapl Component 2')
        plt.colorbar()
        plt.show();        