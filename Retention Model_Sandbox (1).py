#!/usr/bin/env python
# coding: utf-8

# ###  Analysis of the ISA Balance Attrition Data
# @Tanaby Mofrad

# In[1]:


import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

__FILE_LOC = "../train_dataset/traindf.parquet"


# In[2]:


# lets read the data in to a pandas data frame
df_train = pd.read_parquet(__FILE_LOC)


# In[3]:


df_train.columns.to_list()


# In[4]:


print(df_train["is_target"].value_counts())
df_train["is_target"].hist()


# In[5]:


df_reduced = df_train.drop(columns= ["customer_key", "is_target"])


# In[6]:


standardized_data = StandardScaler().fit_transform(df_reduced)


# In[7]:


data_selected= standardized_data[0:1000,:]
labels_selected = df_train["is_target"][0:1000]
model = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=90)
tsne_data = model.fit_transform(data_selected)
tsne_data = np.vstack((tsne_data.T, labels_selected)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# In[8]:


# Maybe there are some clusters???!?! lets do a uniform sampling from data
from numpy.random import uniform
indices = uniform(0,standardized_data.shape[0], 5000)
indices_list = [int(x) for x in indices]
sampled_data = standardized_data[indices_list,]
sampled_target = df_train["is_target"][indices_list] 


# In[9]:


model = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=90)

tsne_data = model.fit_transform(sampled_data)
tsne_data = np.vstack((tsne_data.T, sampled_target)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

sns.FacetGrid(tsne_df, hue="label", size=6)     .map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    
plt.show()
    


# In[ ]:


#### Applying the same concept on targets

df_index = df_train.index[df_train["is_target"]==1].to_list()
print("Size of the file is %d"%len(df_index))
sample_data = standardized_data[df_index]

for perpelxity in range(20,100):
    print(perpelxity)
    model = TSNE(n_components=2, random_state=0, n_iter=1000, perplexity=perpelxity)

    tsne_data = model.fit_transform(sampled_data)
    #tsne_data = np.vstack((tsne_data.T, sampled_target)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2"))

    sns.FacetGrid(tsne_df, size=6)         .map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()


    plt.show()


# In[4]:


### play around with Catboost

import catboost as cb
import gc
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample

import warnings
warnings.filterwarnings('ignore')

#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 10 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round

#CATBOOST PARAMETERS
CB_MAX_DEPTH = 8 #maximum tree depth in CatBoost
OBJECTIVE_CB_REG = 'MAE' #CatBoost regression metric
OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric

#OPTIONAL OUTPUT
BEST_SCORE = 0



def quick_hyperopt(data, labels, num_evals=NUM_EVALS, diagnostic=False):
    """
    Hyper Parameter Optimization
    
    """
        
    print('Running {} rounds of CatBoost parameter optimisation:'.format(num_evals))

    #clear memory 
    gc.collect()

    integer_params = ['depth',
                      #'one_hot_max_size', #for categorical data
                      'min_data_in_leaf',
                      'max_bin']

    def objective(space_params):

        #cast integer params from float to int
        for param in integer_params:
            space_params[param] = int(space_params[param])

        #extract nested conditional parameters
        if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
            bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')
            space_params['bagging_temperature'] = bagging_temp

        if space_params['grow_policy']['grow_policy'] == 'LossGuide':
            max_leaves = space_params['grow_policy'].get('max_leaves')
            space_params['max_leaves'] = int(max_leaves)

        space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']
        space_params['grow_policy'] = space_params['grow_policy']['grow_policy']

        #random_strength cannot be < 0
        space_params['random_strength'] = max(space_params['random_strength'], 0)
        #fold_len_multiplier cannot be < 1
        space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)

        #for classification set stratified=True
        cv_results = cb.cv(train, space_params, fold_count=N_FOLDS, 
                         early_stopping_rounds=25, stratified=True, partition_random_seed=42, plot=True)

        #best_loss = cv_results['test-MAE-mean'].iloc[-1] #'test-RMSE-mean' for RMSE
        #for classification, comment out the line above and uncomment the line below:
        best_loss = cv_results['test-logloss-mean'].iloc[-1]
        #if necessary, replace 'test-Logloss-mean' with 'test-[your-preferred-metric]-mean'

        return{'loss':best_loss, 'status': STATUS_OK}

    train = cb.Pool(data, labels.astype('float32'))

    #integer and string parameters, used with hp.choice()
    bootstrap_type = [ {'bootstrap_type':'Bayesian',
                        'bagging_temperature' : hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
                      {'bootstrap_type':'Bernoulli'}] 

    LEB = ['No', 'AnyImprovement'] #remove 'Armijo' if not using GPU
    score_function = ['Correlation', 'L2', 'NewtonCorrelation', 'NewtonL2']
    grow_policy = [{'grow_policy':'SymmetricTree'},
                   {'grow_policy':'Depthwise'},
                   {'grow_policy':'Lossguide',
                    'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]
    eval_metric_list_reg = ['MAE', 'RMSE', 'Poisson']
    eval_metric_list_class = ['F1','AUC']
    #for classification change line below to 'eval_metric_list = eval_metric_list_class'
    eval_metric_list = eval_metric_list_class

    space ={'depth': hp.quniform('depth', 2, CB_MAX_DEPTH, 1),
            'max_bin' : hp.quniform('max_bin', 254, 254, 1), #if using CPU just set this to 254
            'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0, 5),
            'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),
            'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),
            #'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1), #uncomment if using categorical features
            'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),
            'learning_rate' : hp.uniform('learning_rate', 0.05, 0.25),
            'eval_metric' : hp.choice('eval_metric', eval_metric_list),
            'objective' : OBJECTIVE_CB_CLASS,
            #'score_function' : hp.choice('score_function', score_function), #crashes kernel - reason unknown
            'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),
            'grow_policy': hp.choice('grow_policy', grow_policy),
            'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),# CPU only
            'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
            'od_type' : 'Iter',
            'od_wait' : 25,
            'task_type' : 'CPU',
            'verbose' : 0
        }


    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=num_evals, 
                trials=trials)

    #unpack nested dicts first
    best['bootstrap_type'] = bootstrap_type[best['bootstrap_type']]['bootstrap_type']
    best['grow_policy'] = grow_policy[best['grow_policy']]['grow_policy']
    best['eval_metric'] = eval_metric_list[best['eval_metric']]

    #best['score_function'] = score_function[best['score_function']] 
    #best['leaf_estimation_method'] = LEM[best['leaf_estimation_method']] #CPU only
    best['leaf_estimation_backtracking'] = LEB[best['leaf_estimation_backtracking']]        

    #cast floats of integer params to int
    for param in integer_params:
        best[param] = int(best[param])
    if 'max_leaves' in best:
        best['max_leaves'] = int(best['max_leaves'])

    print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')

    if diagnostic:
        return(best, trials)
    else:
        return(best)
    


# In[ ]:





# In[5]:


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


train_cols = [col for col in df_train.columns if col not in ['customer_key','is_target']]


X_train, X_test, y_train, y_test = train_test_split( df_train[train_cols], df_train["is_target"], test_size=0.33, random_state=42, 
                                                    stratify=df_train["is_target"].values)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, 
                                                  random_state=42, stratify=y_train.values)

# Tobe added: pickle the splits!!!





# Fit model
#model.fit(df_train[train_cols],df_train['is_target'],plot="True")



# In[17]:


model = CatBoostClassifier(iterations=198,
                           learning_rate=0.5,
                           custom_loss = ['AUC','F1'],
                           depth=5)

model.fit(X_train,y_train,
          eval_set=(X_val,y_val),
          plot="True")


# In[ ]:


from catboost.utils import get_fpr_curve, get_fnr_curve
(threshold, fpr) = get_fpr_curve(curve=curve)
(threshold, fnr) = get_fnr_curve(curve=curve)


# In[ ]:


plt.figure(figsize=(16,8))
lw = 2
plt.plot(thresholds, fpr, color='blue', lw=lw, label="FPR", alpha=0.5)
plt.plot(thresholds, fnr, color='green', lw=lw, label="FNR", alpha=0.5)

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xticks(fontsize=16)
plt.grid(True)
plt.ylabel('Error Rate', fon)
plt.xlabel()


# In[ ]:


quick_hyperopt(X_train,y_train)


# In[ ]:




