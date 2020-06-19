# Preprocess Package:

This package is developed to help Data Scientists to accelerate data preprocessing steps.

## Preprocessor Module:

This class is developed in a way to be generalized for different types of datasets. It compiles several functions that preprocess both numeric and categorical data. It includes functions that handle duplicates, identify invalid data (i.e. invalid labels, string values in numeric columns and numeric values in string type columns), impute/drop missing values and scale/normalize the data. 

It relies on 'dask' library for parallel computing and as a result, can handle large datasets. The input parameters are as following:

        numeric_features : list of strings
            A list containing the column labels corresponding to numeric features,
        categorical features: list of strings, default = None
            A list containing the column labels corresponding to categorical features
        target_feature: string
            Column label corresponding to the target feature
        train_df: a Dask Dataframe
            A dataframe containing training data
        test_df: a Dask Dataframe, default = None
            A dataframe containing the testing data
        drop_threshold: int, default = 50
            Percentage of missing values in a feature below which a column will be dropped
        labels: list of strings
            A list containing the expected labels in the target feature
        category_threshold: integer
            Count of categories below which (or equal) a category will be grouped together as 'Other' for a categorical feature
        missing_method: string
            How to handle missing values in a feature, options are 'fill' and 'drop'
                fill: the missing values are imputed with the mean for the numeric features and the mode for categorical features
                drop: the missing values are dropped entirely for both numeric and categorical features
        scale_method: string, default = 'standardscaler'
            Which scaling algorithm to use on the data
                standardscaler: applies Scikit-Learn's StandardScaler
                minmax: applies Scikit-Learn's MinMaxScaler
                robust: applies Scikit-Learn's RobustScaler
        scale_range: tuple(lower_limit, upper_limit)
            Specifies the upper and lower limits of scaling; applicable only for the minmax method only.
        transform_method: string, default = 'yeo-johnson'
            Which transformation algorithm to use on the data
                yeo-johnson: works for both positive and negative values
                box-cox: works for positive values only 

        
