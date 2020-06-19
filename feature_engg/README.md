# Features Package:

This package is developed to help Data Scientists to accelerate Feature Engineering step of their projects. 
This package consists of 2 modules (FeatureEng and FeatureSelect).

## FeatureEng Module:
Creates new feature columns for dataset based on selected aggregations and transformations. Following are the input parameters for this module:

        filepath : str
            the location of dataset. The string could be a URL. 
            Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is expected.
        
        header_type : int, list of int
            Row number(s) to use as the column names, and the start of the data.
            if header_type = 'infer', column names are inferred from the first line of the file, 
            if column names are passed explicitly then the behavior is identical to header=None. 
        
        categorical_col_name :  str, default = None
            The name of categorical columns
        
        label_col_name : str, default = None
            The name of label(target) column of dataframe

        list_agg_primitives : list[str or AggregationPrimitive], default = None
            list of Aggregation Feature types to apply.
                example types of aggreation for ft.dfs:
                ["sum", "std", "max", "skew", "min", "mean", "count",
				"percent_true", "num_unique", "mode"]

        list_trans_primitives : list[str or TransformPrimitive], default = ['multiply_numeric']
            List of Transform Feature functions to apply.
            example types of aggreation for ft.dfs:
            ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]
        For more available perimitives : https://primitives.featurelabs.com/

        max_depth_value : int, default = 1
            Maximum allowed depth of features.

        threshold : int, default = 0.8
            Number between 0 and 1 which determines maximum features correlation limit
	    
	    
### FeatureEng Module Functions:

    separate_features_from_label(self) : separates features from label column
    Returns
    ----------
    label_df : dataframe containing only label(output) column
    label_col_name : column name of label column
    features : dataframe containing only features columns

    numeric_features(self) : separates numeric features, from categorical features and label columns,
    and saves them into 3 separate dataframes.
    Returns
    ----------
    cat_features : dataframe containing only categorical features columns
    numeric_features : dataframe containing only numerical features columns
    categorical_col_name : column name of categorical columns

    new_features(self) : creates new features using current numeric features. Datafarme must only have numeric values.
    Returns
    ----------
    feature_matrix : dataframe containing all the old feature and new synthetized features

    remove_correlated_features(self) : removes highly correlated features
    Returns
    ----------
    feature_matrix : dataframe containing all the features that have correlation less than threshold value

    df_with_new_features(self):
    adds the label(output) column to the dataframe with new features
    Returns
    ----------
    df_new_features : dataframe containing the features and label

## FeatureSelect Module:
Inherits from FeatureEng module and selects features using combination of several methods. This module prepares and saves dataframe with selected features for the next steps of project. Following are the input parameters for this module:

    df_new_features : dataframe 
    	processed dataframe from FeatureEng Parent class
    num_feats : int
    	number of features to be selected
	
### FeatureSelect Module Functions:

    cor_pearson_selector(self) : selects the top n=num_feats features using Pearson’s correlation between featurs and the target. 
    Returns
    ----------
    cor_support : shows if a feature is selected or not. 1 for selected, and 0 for not selected
    cor_feature : list of selected features

    chi_square_selector(self) : selects top n=num_feats features using Chi-Squared method.
    Returns
    ----------
    chi_feature : list of selected features

    log_reg_selector(self) : selects top n=num_feats features using logistic regression method.
    Returns
    ----------
    embeded_lr_feature : list of selected features

    random_forest_selector(self) : selects top n=num_feats features using Random Forest method.
    Returns
    ----------
    embeded_rf_feature : list of selected features

    LGBM_selector(self) : selects top n=num_feats features using Light GBM method.
    Returns
    ----------
    embeded_lgb_feature : list of selected features

    Extra_Trees_selector(self) : selects top n=num_feats features using Extra Tree Classifier.
    Returns
    ----------
    extra_trees_feature : list of selected features

    combine_selector(self) : selects top n=num_feats features using combinations of methods.
    Returns
    ----------
    feature_selection_df.head(num_feats) : dataframe containing the top features with their 
    number of votes for each feature selection method.

    save_df_selected_columns(self) : saves final dataset with top n=num_feats selected features.
    Returns
    ----------
    df_selected_columns : dataframe containing the top features and label to be used for next steps.



 
 
