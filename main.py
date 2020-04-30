# -*- coding: utf-8 -*-

from mlpipeline.CatboostML import CatboostModel
from mlpipeline.xgb_class import XGBoostModel
from mlpipeline import lgbmclass as lgbc
import pandas as pd
import logging
import configparser
import extentions.utilities as utilities
import os



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
    optimization = 

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



    ######   AHMAD #####   XGBOOST
    xgb_model = XGBoostModel(train=train_data, test=test_data, target_feature=0, max_evals = 10,
                            n_fold=5, num_boost_rounds=100, early_stopping_rounds=10,
                            seed=42, GPU=False)
    xgb_model.train_models(optim_type='hyperopt')

    ##### KShitij   #### LightGBM 
    obj = lgbc.Lgbmclass(train_X, train_y)
    obj.parameter_tuning('optuna_space')
    obj.train(test_X, test_y)

    #### Tanaby #### Apply the test set and get the model evaluation results





if __name__ == "__main__":
    args = _get_args()
    log_filename = args.log_path
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(filename=log_filename, filemode='w+', level=logging.INFO)
    main()









