# -*- coding: utf-8 -*-

from mlpipeline.catboost_class import Ctbclass
from mlpipeline.xgb_class import XGBoostModel
from mlpipeline import lgbmclass as lgbc
from mlpipeline.evaluations import Model_Evaluation
import pandas as pd
import logging
import configparser
import extensions.utilities as utilities
import os
import warnings
from sklearn.exceptions import DataConversionWarning
import argparse
import json
import pickle as pkl

pd.options.mode.chained_assignment = None
warnings.filterwarnings(action='ignore', category=DataConversionWarning) 


def _get_args():
    """Get input arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--categorical_columns",
                        default="data/categorical_features.json",
                        help="path to categorical data.",
                        type=str)

    parser.add_argument("--save_path",
                        default="model",
                        help="path to encoder and model.",
                        type=str)

    parser.add_argument("--data_path",
                        default="data/higgs_data.csv",
                        help="Path to  higgs_data",
                        type=str)

    parser.add_argument("--config_path",
                        default="config.json",
                        help="path to configfile  path.",
                        type=str)

    parser.add_argument("--result_path",
                        default="results/",
                        help="path to result path.",
                        type=str)

    parser.add_argument("--log_path",
                        default="log/higgs_project.log",
                        help="path to log path.",
                        type=str)

    parser.add_argument("--featureset_max_num",
                        default="17",
                        help="maximum number of created features.",
                        type=int)

    parser.add_argument("--algorithm",
                        default="lgb",
                        help="The algorithm to use for training",
                        type=str,
                        choices=['xgb', 'ctb', 'lgb'])

    parser.add_argument("--optimization",
                        default="hyperopt",
                        help="The optimization to use",
                        type=str,
                        choices=['hyperopt', 'optuna', 'random_search'])

    return parser.parse_args()

def main():
    """
    The main function, reads all parms, args and runs the train/test

    """
    args = _get_args()
    numerical_f_path = args.numeric_columns
    categorical_f_path = args.categorical_columns
    data_path = args.data_path
    config_path = args.config_path
    result_path = args.result_path
    save_path = args.save_path
    optimization = args.optimization
    algorithm = args.algorithm

	# Read the configuration file
    # config = json.load(open(config_path, 'r'))
    

    df_data = utilities.load_data(data_path, sample_rate=0.01)
    logging.info('Dataframe of shape {} loaded'.format(df_data.shape))

    # preprocess data
    # df_data = utilities.preprocess_train_test()

    # create train-test dataset
    X_train, X_test, y_train, y_test = utilities.create_test_train(df_data)

    # start with model training:

    if algorithm == "ctb":
    
        model= Ctbclass(X_train, y_train)
        model.train(hyperparameter_optimizer=optimization)
        model.test(X_test, y_test) 
        predictions = model.predictions

    elif algorithm == "xgb":

        model = XGBoostModel(X_train, y_train, max_evals=50, n_fold=5, 
                        num_boost_rounds=100, early_stopping_rounds=10,
                        seed=42, GPU=False)
        model.train(optim_type=optimization)
        model.test(X_test, y_test)
        predictions = model.predictions

    elif algorithm == 'lgb':

        model = lgbc.Lgbmclass(X_train, y_train)
        model.train(optimization)
        model.test(X_test, y_test)
        predictions = model.pred

    #### Apply the test set and get the model evaluation results

    me = Model_Evaluation() 
    me.set_label_scores(predictions,y_test)

    roc_filename  = "roc_" + algorithm + "_" + optimization +".png"
    roc_filename = os.path.join("figs", roc_filename)
    pr_filename  = "pr_" + algorithm + "_" + optimization +".png"
    pr_filename = os.path.join("figs", pr_filename)
    fpr_fnr_filename = "fpr_fnr_" + "_" + algorithm + "_" + optimization +".png"
    fpr_fnr_filename = os.path.join("figs", fpr_fnr_filename)

    results = me.get_metrics(roc_filename,
                   pr_filename,
                   fpr_fnr_filename,
                   algorithm)

    if  not results:
        raise Exception("no results generated! please check!")
    else:  ### Print the results #### 
        logging.info("results generated, will be printing the results")
        print ("*" * 100)
        print ("Results obtained from %s" %(algorithm))
        print ("   PR_AUC   :", results['pr-auc'])
        print( " Classification Report  \n", results['class_report'])
        print (" Confusion Matrix   \n", results['conf_metrics'])
        print (" ROC AUC   ",results['roc_auc'])
        print ("*" * 100)
        pkl.dump(results, open(algorithm +"_"+ optimization, "wb"))

if __name__ == "__main__":
    args = _get_args()
    log_filename = args.log_path
    
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    try:
        os.makedirs("figs")
    except OSError as exc:
        print('saving results in figs folder')
        
    logging.basicConfig(filename=log_filename, filemode='w+', level=logging.INFO)
    main()