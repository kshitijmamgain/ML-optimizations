# -*- coding: utf-8 -*-

#from mlpipeline.catboost_class import Ctbclass
#from mlpipeline.xgb_class import XGBoostModel
from mlpipeline import lgbmclass
from mlpipeline.evaluations import Model_Evaluation
import pandas as pd
import logging
import configparser
import extensions.utilities as utilities
import os
import warnings
from sklearn.exceptions import DataConversionWarning
import argparse
import datetime
import pickle as pkl
from timeit import default_timer as timer
import yaml
import sys

pd.options.mode.chained_assignment = None
warnings.filterwarnings(action='ignore', category=DataConversionWarning) 
    
def _get_args():
    """Get input arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config_path",
                        default="config.yml",
                        help="path to configfile  path.",
                        type=str)

    parser.add_argument("--result_path",
                        default="results",
                        help="path to result path.",
                        type=str)

    parser.add_argument("--log_path",
                        default="log/"+'{:%Y%m%d-%H:%M:%S}'.format(datetime.datetime.now())+"-project.log",
                        help="path to log path.",
                        type=str)

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

    config_path = args.config_path
    result = args.result_path
    optimization = args.optimization
    algorithm = args.algorithm
    home = os.getcwd()
    # load data file path
    with open(config_path) as file:
        config = yaml.safe_load(file)
    train_data_path = home+config['path']['train_data']
    test_data_path = home+config['path']['test_data']
    target_label = config['target']['label']
	# Read the configuration file
    # config = json.load(open(config_path, 'r'))

    
    #loading training and testing data into dataframes
    df_train = utilities.load_data(path=train_data_path, sample_rate=None)
    logging.info('Train Dataframe of shape {} loaded'.format(df_train.shape))
    df_test = utilities.load_data(path=test_data_path, sample_rate=None)
    logging.info('Test Dataframe of shape {} loaded'.format(df_test.shape))

    # create X and y
    X_train, y_train = utilities.create_xy(df=df_train, target=target_label)
    X_test, y_test = utilities.create_xy(df=df_test, target=target_label)


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
        
        start = timer()
        model = lgbmclass.Lgbmclass(X_train, y_train)
        model.train(optimization, device='cpu')
        train_time = timer() - start
        logging.info('Train time {} with {} optimization: {} seconds'.format(algorithm, optimization, train_time))
        model.test(X_test, y_test)
        test_predictions=model.test_prediction
        print('test_prediction', test_predictions)
        logging.info('Train time with best parameters for {} with {} optimization: {} seconds'.format(algorithm, optimization, model.best_time))
        #predictions = model.pred
        train_predictions=model.train_prediction
        print('train_prediction', train_predictions)

    #### Apply the test set and get the model evaluation results

    roc_filename  = "roc_" + algorithm + "_" + optimization +".png"
    roc_filename = os.path.join(result, roc_filename)
    pr_filename  = "pr_" + algorithm + "_" + optimization +".png"
    pr_filename = os.path.join(result, pr_filename)
    fpr_fnr_filename = "fpr_fnr_" + "_" + algorithm + "_" + optimization +".png"
    fpr_fnr_filename = os.path.join(result, fpr_fnr_filename)

    for predictions, actual, mode in [(train_predictions, y_train, 'train'), (test_predictions, y_test, 'test')]:
        me = Model_Evaluation() 
        me.set_label_scores(predictions, actual)
        results = me.get_metrics(roc_filename,
                       pr_filename,
                       fpr_fnr_filename,
                       algorithm)
        if  not results:
            raise Exception("no results generated! please check!")
        else:  ### Print the results #### 
            logging.info("results generated, will be printing the results")
            print ("*" * 100)
            print ('Mode: '+ mode)
            print ("Results obtained from %s" %(algorithm))
            print ("   PR_AUC   :", results['pr-auc'])
            print( " Classification Report  \n", results['class_report'])
            print (" Confusion Matrix   \n", results['conf_metrics'])
            print (" ROC AUC   ",results['roc_auc'])
            print ("*" * 100)
            pkl.dump(results, open(algorithm +"_"+ optimization +"_" + mode, "wb"))

if __name__ == "__main__":
    args = _get_args()
    log_filename = args.log_path
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    #sys.stdout = open(log_filename, 'w')
    result = args.result_path    
    try:
        os.makedirs(result)
    except OSError as exc:
        print('saving results in figs folder')
        
    logging.basicConfig(filename=log_filename, filemode='w+', level=logging.INFO)
    main()