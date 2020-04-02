import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
sns.set(style='ticks')
np.set_printoptions(precision=2)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def countNull(df):
    # returns a dataframe with number and percent of null values in every feature
    # output ignores features with zero null values

    df = df.copy()
    N = df.shape[0]
    df_null = df.isnull().sum()[df.isnull().sum() > 0].to_frame('Absolute')
    df_null['Percent'] = df_null.Absolute * 100 / N

    return df_null

def tooSkewedFeatures(df, tol, test, ignore_unskewed=True):
    # returns 1. columns # 2. their respective moments
    # perform the supplied test and also output the 3. pvalue
    # output is a dataframe
    # if ignore_unskewed=True, columns which have skewness below tol are ignored

    # compute skews
    means = []
    stds = []
    skews = []
    kurts = []
    pvals = []

    for col in df.columns:
        # drop null values
        values = df[col].dropna().values

        # perform supplied test
        if test is None:
            pval = None
        else:
            pval = test(values)

        # append
        means.append(values.mean())
        stds.append(values.std())
        skews.append(stats.skew(values))
        kurts.append(stats.kurtosis(values))
        pvals.append(pval)

    # prepare output dataframe
    to_df = {'Mean': means, 'Std dev': stds, 'Skewness': skews, 'Ex kurtosis': kurts, 'P-value': pvals}
    results = pd.DataFrame(to_df, index=df.columns)
                 
    if ignore_unskewed:
        # dont output columns with small skewness
        results = results[results.Skewness >= tol]

    if test is None:
        # if no test was supplied, drop the (fake) pvalue column
        results.drop('P-value', axis=1, inplace=True)

    return results

class EDA():
    def __init__(self, target_label, is_target_cat, alpha_pvalue=0.05,
                 norm_test='KS', norm_test_func=None, corr_tol=0.60, corr_tol_low=0.10,
                 corr_how='spearman', make_plots=True, cont_tol=1.0,
                 mode_thresh=40.0, imbalance_factor=2, VIF_thresh=5.0,
                 skew_tol=1.0, class_thresh=6, marker_size=15,
                 marker_alpha=0.5, plot_cols=2, figsize=(8, 4), colorblind=True):
        self.target = target_label
        self.alpha = alpha_pvalue
        self.norm_test = norm_test
        self.norm_test_func = norm_test_func
        self.make_plots = make_plots
        self.corr_tol = corr_tol
        self.corr_tol_low = corr_tol_low
        self.corr_how = corr_how
        self.is_target_cat = is_target_cat
        self.make_plots = make_plots
        self.marker_size = marker_size
        self.marker_alpha = marker_alpha
        self.figsize = figsize
        self.cont_tol = cont_tol
        self.mode_thresh = mode_thresh
        self.imbalance_factor = imbalance_factor
        self.skew_tol = skew_tol
        self.plot_cols = plot_cols
        self.class_thresh = class_thresh
        self.colorblind = colorblind
        self.VIF_thresh = VIF_thresh

    def doEDA(self, df):

        df = df.copy()

        # plot dimension
        width = self.figsize[0]
        height = self.figsize[1]
        
        print("\nHEAD")
        print(df.head())

        # compute/print number of features and samples
        print("\nSHAPE")
        N = df.shape[0]
        M = df.shape[1]
        print('Number of samples', N)
        print('Number of columns', M)

        # separate feature types (categorical and numerical)
        categ_cols = df.dtypes[(df.dtypes == 'O') | (
            df.dtypes == 'category')].index.values
        num_cols = [col for col in df.columns
                    if col not in categ_cols]

        # compute/print number of missing values
        print('\nMISSING VALUES')
        self.missing_values_df_ = countNull(df)
        if self.missing_values_df_.empty:
            print('No missing values')
        else:
            print(self.missing_values_df_)
            
        print('\nSCATTERPLOTS OF FEATURES VS INDEX')
        for col in df.drop(self.target, axis=1):
            plt.scatter(x=df.index, y=df[col], label=col)
            plt.xlabel('index')
            plt.ylabel(col)
            plt.title('{} vs index'.format(col))
            plt.show()

        if self.is_target_cat:
            # make count plot to show target category populations
            print('\nTARGET DISTRIBUTION \nNumbers displayed on bars are counts relative to the least common category')
            plt.figure(figsize=self.figsize)
            g = sns.countplot(x=self.target, data=df)
            # add text that displays relative size of bars
            heights = np.array([p.get_height() for p in g.patches],
                               dtype='float')
            y_offset = 0.05*heights.min()  # additive
            # rescale heights relative to smallest bar
            heights = 100.0*heights/heights.min()
            for i, p in enumerate(g.patches):
                g.text(p.get_x(),
                       p.get_height() + y_offset,
                       '{0:.0f}'.format(heights[i]),
                       ha="left")
            plt.show()
        else:
            # make distplot for continuous target
            plt.figure(figsize=self.figsize)
            bins = int(len(df[self.target]) / 25)
            g = sns.distplot(df[self.target],
                             bins=bins)
            plt.show()
            
        # find columns who are too skewed
        # too skewed means skewness is over self.skew_tol
        print('SKEWNESS')
        print('Features which have skewness above {0}'.format(self.skew_tol))

        # define normality tests
        if self.norm_test == None:
            normality_test = None
        elif self.norm_test == 'KS':  # KS test
            normality_test = lambda x: (stats.kstest(x, 'norm')[1])
            print('(P-values are for Kolmogorov-Smirnov test of normality)\n')
        elif self.norm_test == 'SW':  # SW test
            normality_test = lambda x: stats.shapiro(x)[1]
            print('(P-values are for Shapiro-Wilk test of normality)\n')
        elif self.norm_test == 'custom':
            normality_test = self.norm_test_func
            print('(P-values are for supplied test of normality)\n')
        else:
            normality_test = None
            print('No normality test performed')
            

        to_search_from = num_cols

        self.too_skewed_ = tooSkewedFeatures(df[to_search_from],
                                             tol=self.skew_tol,
                                             test=normality_test)
        print(self.too_skewed_)
        
        print('\nHISTOGRAMS OF FEATURES')
        for col in df.drop(self.target, axis=1):
            plt.hist(df[col], label=col)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.title('Distribution of {}'.format(col))
            plt.show()
           
        # Find correlation between the features and features with target
        print('PAIRWISE FEATURE CORRELATION, TOP 10')
        correlations = {}
        columns = df.drop('id', axis=1).columns.tolist()
            
        for col_a, col_b in itertools.combinations(columns, 2):
            correlations[col_a + '\nvs\n' + col_b] = stats.pearsonr(df.drop('id', axis=1).loc[:, col_a], 
                                                                df.drop('id', axis=1).loc[:, col_b])
            
        result = pd.DataFrame.from_dict(correlations, orient='index')
        result.columns = ['correlation', 'p-value']
        print(result.sort_values(by='correlation', ascending=False).head(10))
            
        #plot heatmap of correlations
        corr = df.drop('id', axis=1).corr()
        plt.figure(figsize=(20,20))
        g=sns.heatmap(corr,annot=True,cmap="RdYlGn")
