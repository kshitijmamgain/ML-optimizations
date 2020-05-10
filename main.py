# -*- coding: utf-8 -*-

from mlpipeline.CatboostML import CatboostModel
from mlpipline.evaluation import Model_Evaluation
#from mlpipeline import XGboostModel 
import pandas as pd
import logging
import configparser
import extentions.utilities as utilities
import os
import pickle as pkl



pd.options.mode.chained_assignment = None
warnings.filterwarnings(action='ignore', category=DataConversionWarning) 


def _get_args():
    """Get input arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--numeric_columns",
                        default="data/numerical_features.json",
                        help="path to numeric data.")

    parser.add_argument("--categorical_columns",
                        default="data/categorical_features.json",
                        help="path to categorical data.")

    parser.add_argument("--save_path",
                        default="model",
                        help="path to encoder and model.")

    parser.add_argument("--data_path",
                        default="data/higgs_data.csv",
                        help="Path to  higgs_data")

    parser.add_argument("--config_path",
                        default="config.json",
                        help="path to configfile  path.")
    parser.add_argument("--result_path",
                        default="results/",
                        help="path to result path.")

    parser.add_argument("--log_path",
                        default="log/higgs_project.log",
                        help="path to log path.")
    parser.add_argument("--featureset_max_num",
                        default="17",
                        help="maximum number of created features.")

    parser.add_argument("--algorithm",
                        default="catboost",
                        help="The algorithm to use for trainging")

    parser.add_argument("--optimization",
                        default="hyperopt"
                        help="The optimization to use")

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
	config = json.load(open(config_path, 'r'))
    

	df_data = utilities.load_data(data_path)
	logging.info(f'Dataframe of shape {df.shape} loaded')

    # preprocess data
    df_data = utilities.preprocess_train_test()

    # create train-test dataset
    train_data, test_data = utilities.create_test_train(df_data)

    # start with model training:


    #### Tanaby ##### Work on config and request parser
    ##### Tanaby add the logging info 




    ####### SASHA ######## ADD the CATBOOST Class (Vanilla class for training)

    if algorithm == "ctb":
    
        model= Ctbclass(x_train, y_train, 'GPU', 'random')
        model.train(x_test,y_test)
        predictions =  model.test (X_test,y_test)

    elif algorithm == "xgb":

        model = XGBoostModel(train=train_data, test=test_data, target_feature=0, max_evals = 10,
                            n_fold=5, num_boost_rounds=100, early_stopping_rounds=10,
                            seed=42, GPU=False)
        model.train_model(optim_type='hyperopt')
        predictions =  model.test (X_test,y_test)


    else:

        model = lgbc.Lgbmclass(train_X, train_y)
        model.parameter_tuning('optuna_space')
        model.train(test_X, test_y)
        predictions =  model.test (X_test,y_test)


    #### Tanaby #### Apply the test set and get the model evaluation results


    me = Model_Evaluation() 
    me.set_label_scores(predictions,y_test)

    roc_filename  = "roc_" + algorithm + "_" + optimization +".jpg"
    roc_filename = os.path.join("figs",roc_filename)
    pr_filename  = "pr_" + algorithm + "_" + optimization +".jpg"
    pr_filename = os.path.join("figs",pr_filename)
    fpr_fnr_filename = "fpr_fnr_" + "_" + algorithm + "_" + optimization +".jpg"
    fpr_fnr_filename = os.path.join("figs",fpr_fnr_filename)

    results = me.get_metrics(roc_filename,
                   pr_filename,
                   fpr_fnr_filename,
                   algorithm)

    if  not results: raise Exception("no results generated! please check!")
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
    os.makedirs("figs")

    logging.basicConfig(filename=log_filename, filemode='w+', level=logging.INFO)
    main()









