##Some requirements below are redundant!!!
import pandas as pd
import sklearn
import numpy as np
from matplotlib import pyplot as plt
import pandas.util.testing as tm
from numpy.random import uniform
import catboost as cb
import gc
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.stochastic import sample
import warnings
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score,auc, roc_auc_score, roc_curve,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import csv
from timeit import default_timer as timer
import random
from catboost import CatBoostClassifier
from catboost import *
from numpy.random import RandomState

##Data
__FILE_LOC = "c:/users/hajyhass/downloads/train_test_files_sample.csv"
df_train = pd.read_csv(__FILE_LOC,header=0)
df_reduced = df_train.drop(columns= ["0"])
data_selected= df_reduced[:][:]
labels_selected = df_train["0"][:]
indices = uniform(0,df_reduced.shape[0], 5000)
indices_list = [int(x) for x in indices]
sampled_data = df_reduced.iloc[indices_list,]
sampled_data=sampled_data.drop(columns='Unnamed: 0')
sampled_target = df_train["0"][indices_list] 
warnings.filterwarnings('ignore')
# making a smaller df for quick testing
df_s, _ = train_test_split(df_train, random_state = 30, train_size = 0.1)
train_X = sampled_data
train_y = sampled_target



#Configure Defaults
MAX_EVALS = 5
NUM_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 25
SEED = 47
#GLOBAL HYPEROPT PARAMETERS ->These must be in Configure file already
#NUM_EVALS = 5 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round
#CATBOOST PARAMETERS
CB_MAX_DEPTH = 16 #maximum tree depth in CatBoost
#OBJECTIVE_CB_REG = 'MAE' #CatBoost regression metric
#OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric


##The Object Orineted below is modified on Kshitij's original script
class Parameter_Tuning:

    '''The Parameter_Tuning Class, optimizes via Hyperopt.The class is composed of different functions including:
    __init__() that initializes the input arguments,
    hyperopt_space() which runs a global optimization over a pre-defined parameter speace,
    ctb_cv() which is the main catboost technology'''
    '''The Parameter_Tuning Class, needs other functions including:
    tuner() to feed optimal values into,
    predict() to predict the results based on most optimized parameters,
    the metric() to evaluate the results
    same pattern must be repeated for Optuna'''

    def __init__(self, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train
        self.train_set = cb.Pool(self.x_train, self.y_train)

    def hyperopt_space(self, fn_name, space, algo, trials):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=MAX_EVALS,
                          trials=trials, rstate=np.random.RandomState(50))
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}
        print(result, trials)
        return result, trials

    def ctb_cv(self, params):
        
        """ What are we achieving below?
        Sometimes we have a dictionary inside which there are different types of input, for example one choice
        is a 1-key dict and the other choices are 2-key dicts. Therefore, we msut define a conditional statement
        to identify which type of dictionary we are dealing with, and if it is of inconsistent type,then we normalize it"""
        
        """After removing the inconsistencies of the params_space inputs, we split the data via cross-validation and 
        feed them all into the Catboost using the 
        the results of each evaluation
        """

        # Extract the bootstrap_type

        if params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
            print(params['bootstrap_type'])
            params['bagging_temperature'] = params['bootstrap_type']['bagging_temperature']
            print(params['bagging_temperature'])
            params['bootstrap_type'] = params['bootstrap_type']['bootstrap_type']
            print(params['bootstrap_type'])
        else:
            params['bootstrap_type'] = params['bootstrap_type']['bootstrap_type']
            print(params['bootstrap_type'])

        if params['grow_policy']['grow_policy'] == 'Lossguide':
            params['max_leaves'] = params['grow_policy']['max_leaves']
            print(params['max_leaves'])
            params['grow_policy'] = params['grow_policy']['grow_policy']
            score_function = 'NewtonL2'
            # when 'grow_policy' is set to 'lossguide', the only possible score_function is 'NewtonL2'
        else:
            params['grow_policy'] = params['grow_policy']['grow_policy']
            print(params['grow_policy'])

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['l2_leaf_reg', 'depth', 'border_count']:
            params[parameter_name] = int(params[parameter_name])

        cv_results = cb.cv(self.train_set, params, fold_count=N_FOLDS, num_boost_round=NUM_BOOST_ROUNDS,
                           early_stopping_rounds=EARLY_STOPPING_ROUNDS, stratified=True, partition_random_seed=42,
                           plot=True)

        # Extract the best score
        # performance measures are not specific to regression or classification (except for R^2)
        best_score = np.max(cv_results['test-F1-mean'])
    

        # Loss must be minimized
        loss = 1-best_score

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}


hyperopt_space = {
    'l2_leaf_reg': hp.qloguniform('l2_leaf_reg', 0, 2, 1),
    'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
    'depth': hp.quniform('depth', 1, CB_MAX_DEPTH, 1),
    'loss_function': hp.choice('loss_function', ['Logloss']), # RMSE and #MAE and Poisson for regression
    'border_count': hp.quniform('border_count', 32, 255, 1),
    'bootstrap_type': hp.choice('bootstrap_type', 
        [{'bootstrap_type': 'Bayesian', 'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
         {'bootstrap_type': 'Bernoulli'}]),
    'grow_policy': hp.choice('grow_policy', 
        [{'grow_policy': 'SymmetricTree'}, {'grow_policy': 'Depthwise'},
         {'grow_policy': 'Lossguide', 'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]),
    'score_function': hp.choice('score_function', ['Cosine']),
    # The score type used to select the next split during the tree construction
    # 'score_function' = ['Correlation', 'L2', 'NewtonCorrelation', 'NewtonL2','LOOL2','Cosine','SatL2','SolarL2']
    # NewtonL2 is the only possible choice for Lossguide
    # The other choices explicitely belong to GPU, only 'Cosine' applies to CPU
    'eval_metric': hp.choice('eval_metric', ['F1']),
    # Eval_metric helps with detecting overfitting
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 50, 1),
    'random_strength': hp.loguniform('random_strength', np.log(0.005), np.log(5)),
    'rsm': hp.uniform('rsm', 0.1, 1),
    # RSM :Random subspace method. The percentage of features to use at each split selection
    'od_type': hp.choice('od_type', ['IncToDec', 'Iter']),
    'task_type': 'CPU',
    # 'thread_count':-1
    'leaf_estimation_backtracking': hp.choice('leaf_estimation_backtracking', ['No', 'AnyImprovement']),
    #'boosting_type':hp.choice('boosting_type',['Ordered','Plain']), depends on symmetricality of the grow_policy
    #Ordered for small dataset requiring high accuracy, Plain for large data sets requiring processing speed
    
}

obj = Parameter_Tuning(train_X, train_y)
ctb_ho = obj.hyperopt_space(fn_name='ctb_cv', space=hyperopt_space, algo=tpe.suggest, trials=Trials())

# Comprehensive Explaination:
# We optimize based on loss_function;Logloss, CrossEntropy,F1,Precision,Recall,AUC, accuracy, etc for classification
# F1,Precision,Recall,AUC, accuracy,TotalF1,MultiClass for multiclass
# RMSE, Poisson, MAE, etc for regression, the entire list is here:
# https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
# Every boosting_round the algo gets feedback from loss_function and improves the next step, meanwhile, early stoppping
# ensures that training and test (eval) results have not diverged,
# In this code I have num_boost_round (or iterations) set to 200 so by default 200 trees will 
# be built unless 25 consecutive divergences occur between the train and eval_set, because early_stopping_rounds=25
# the metric to decide the early_stopping_rounds (overfit detection) is by default "logloss", but
# via eval_metric I can include other forms such as F1
# Boosting is a method which builds a prediction model as an ensemble of weak learners (trees). F(t)=Sum f(t)i
# In our case, f(t) is a decision tree. Trees are built sequentially (200 of them) and each next tree is built to
# approximate negative gradients of the loss function at predictions of the current ensemble:
# Thus, it performs a gradient descent optimization of the loss_function.
# The quality of the gradient approximation is measured by the score_function.
# Let's suppose that it is required to add a new tree to the ensemble, 
# A score function is required in order to choose between candidate trees. 
# how does it achieve that? By applying an optimization of our choice on the aggregate leaves of the tree 
# https://catboost.ai/docs/concepts/algorithm-score-functions.html
# the tree has its own specific parameters such as  
# 1:rsm Random subspace method. The percentage of features to use at each split selection
# 2:min_data_in_leaf:
# 3:random_strength: The amount of randomness to use for scoring splits.The score indicates how much adding this split will
# improve the loss function for the training dataset. 
# 4:border_count: The number of splits for numerical features. CPU:254 and GPU 254 for top quality training
# 5:grow_policy: symmetrical by default unless specified. Default is 10 times faster
# 5a:Symmetric: built level by level to max depth, where lower leaves are split by the same "CONDITION" as upper leaves
# 5b:Depthwise:built level by level to max depth, where each leaf is split by condition with the best loss improvement.
# 5c:Lossguide: built leaf by leaf until the specified maximum number of leaves is reached,on each iteration,
# non-terminal leaf with the best loss improvement is split.
# 6:depth: max 16 for CPU for any loss_functions. Max 16 for GPU except for a few loss_functions
# https://catboost.ai/docs/search/?query=min_data
# 7:leaf_estimation_backtracking: Reduce the descent step when the loss function value is smaller 
# than it was on the previous step, or do nothing
# 8:od_type: Over detection. 
# 8a:iter: Before building each new tree, CatBoost checks the number of iterations
# since the iteration with the optimal loss function value. The model is considered overfitted 
# if the number of iterations exceeds the value specified in the training parameters.
# 8b:IncToDec: Before building each new tree, CatBoost checks the resulting loss change on the validation dataset. 
# is triggered if the Threshold value set in the starting parameters is greater than CurrentPValue
# https://catboost.ai/docs/concepts/overfitting-detector.html#overfitting-detector


# We will do 5 evaluations in this code, So, 5 forests each with 200 tress. 
# Think of each evaluation as a seperate bowl we try to optimize
# We choose a performance measure such as F1,TotalF1,AUC alongside Logloss and get the mean of that metric for each bowl
# ?The choice of eval-metric and performance_measure are mysteriously or reasonably constrained to be identical?
# We identify which bowl has produced the best result, according to the performance criteria
# Return that optimal bowl with all its specific parameters
# The hyperopt's objective is always based on minimization
# Let's say my performance metric is F1, AUC or such which impplies the better the model, the higher those values should be
# Since hyperopt minimizes, I have to write the objective function as 1-F1, this way the objective function will take in
# the largest F1 possible to minimize the 1-F1 the most, hence by definition I will select the bowl with highest F1


# XGboost,Catboost,LGBM all get used for heterogenous data, data which has tables of independent attributes, spam,fraud,
# Neural Nets are used for homogenous data, where input features are related to each other .eg. image, voice,etc
# Gradientboosting builds iterative decision trees
# Catboost splits the tree into 2 branches and it can also handle both numerical and cateorical data
# With larger data and expanded features, Catboost must outperform other algorithms on CPU, and even more on GPU

#Doubts:
## num_boost_rounds Vs od_type exactly identical?
## min_data_in_leaf? 


















