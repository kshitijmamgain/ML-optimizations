import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample
from sklearn.metrics import log_loss
from timeit import default_timer as timer
import random
import seaborn as sns
import shap


class XGB_Higgs():

  """
  Executes methods that split a dataset into training and testing sets, runs an 
  XGBoost classifier with cross-validation, and attempts to optimize the 
  hyperparameters using three different methods: HyperOpt, Optuna and 
  RandomSearch.
  """

  def __init__(self, filename, max_evals, n_fold, seed):

    """
    Initializes an instance of the XGB_Higgs Class

    Parameters
    ----------
    filename: string
      A path containing the data file to be explored
    max_evals: int
      The number of trials to run the optimization algorithms
    n_fold: int
      Number of cross-validation folds to use during validation

    """
    if not isinstance(filename, str):
      raise TypeError('Filepath must be a string')
    if not isinstance(max_evals, int):
      raise TypeError('Number of evaluations must be an integer')
    if not isinstance(n_fold, int):
      raise TypeError('Number of cross-validation folds must be an integer')

    self.filename = filename
    self.data = pd.read_csv(self.filename)
    self.data = self.data.drop(['Unnamed: 0'],axis=1)
    self.max_evals = max_evals
    self.nfold = n_fold
    self.x_train, self.x_test = None, None
    self.y_train, self.y_test = None, None
    self.best_hyperopt_params = None
    self.optuna_params = None
    self.random_search_params = None
    self.best_hyperopt_model = None
    self.best_optuna_model = None
    self.best_random_search_model = None
    self.output = None
    self.dtrain, self.dtest = None,None
    self.seed = seed
    self.trained = False
    self.tested = False

  def split_data(self):
    
    """
    Splits the data into a training and testing test

    """
    x = self.data.iloc[:, 1:22]
    y = self.data.iloc[:, 0]
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.3, random_state = self.seed)
    self.dtrain = xgb.DMatrix(self.x_train, label = self.y_train)
    self.dtest = xgb.DMatrix(self.x_test)


  def cross_validation(self, space):

    """
    Applies cross_validation for the XGBoost algorithm

    Parameters
    ----------
    space: dict
      A dictionary containing the parameters to use when training the model
    
    Returns
    -------
    results: dict
      A dictionary containing the following attributes:
        'loss': the mean value of the loss function across n-fold
         cross-validations
        'variance':  the variance of the loss function across n-fold
        cross-validation
        'params': the parameters used to train the model
        'time': the time it took to run the model
    """

    if not isinstance(space, dict):
      raise TypeError('Parameters must be provided as a dictionary')

    start = timer()
    cv_results = xgb.cv(space, 
                      self.dtrain,
                      num_boost_round = space["n_estimators"],
                      verbose_eval = True,
                      nfold = self.nfold,
                      metrics = space["eval_metric"],
                      seed = self.seed)
    end = timer()
    cv_score = cv_results['test-logloss-mean'].iloc[-1]
    cv_var = (cv_results['test-logloss-std'].iloc[-1])**2

    results = {'loss': cv_score, 'variance': cv_var, 'params': space, 'time': end - start}

    return results



  def hyperopt_tuning(self):

    """
    Applies the HyperOpt algorithm to tune the hyperparameters of the XGBoost
    model and stores the results in a dataframe

    Returns
    -------
    best: dict
      A dictionary containing the set of parameters that resulted in the best
      value of the loss function using the HyperOpt optimization algorithm.
    """
    print("Starting HyperOpt hyperparameter tuning...")
    self.hyperopt_results = pd.DataFrame(columns = ['params', 'loss', 'variance', 'time'])

    def hyperopt_params():
      
      """
      Defines the hyperparameter search space for the HyperOpt algorithm

      Returns
      ----------
      params: dict
        A dictionary containing the search-space for the Hyperopt optimization 
        algorithm
      """

      grow_policy = [{"grow_policy": "depthwise"},

                           {"grow_policy": "lossguide",
                            "max_leaves": hp.quniform("max_leaves", 2, 32, 1)}]

      booster = [{"booster": "gbtree"},

                       {"booster": "dart",
                        "sample_type":
                            hp.choice("sample_type", ["uniform", "weighted"]),
                        "normalize_type":
                            hp.choice("normalize_type", ["tree", "forest"]),
                        "rate_drop":
                            hp.uniform("rate_drop", 0, 1),
                        "skip_drop":
                            hp.uniform("skip_drop", 0, 1)}]

      params = {
                "silent": 1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "booster": hp.choice("booster", booster),
                "reg_lambda": hp.quniform("reg_lambda", 1, 2, 0.1),
                "reg_alpha": hp.quniform("reg_alpha", 0, 10, 1),
                "verbosity": 2,
                "n_estimators":
                    hp.choice("n_estimators", np.arange(5, 10, 1, dtype=int)),
                "max_depth":
                    hp.choice("max_depth", np.arange(1, 14, dtype=int)),
                "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
                "gamma": hp.quniform("gamma", 0.5, 1.0, 0.05),
                "grow_policy": hp.choice("grow_policy", grow_policy),
                "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
                "subsample": hp.quniform("subsample", 0.7, 1, 0.05),
                "colsample_bytree":
                    hp.quniform("colsample_bytree", 0.7, 1, 0.05)
                }

      return params
      
    space = hyperopt_params()

    def hyperopt_objective(space):

      """
      Defines the objective function to be optimized by the HyperOpt algorithm

      Parameters
      ----------
      space: dict
        A dictionary of parameters to use for each trial of the HyperOpt
        optimization algorithm

      Returns
      -------
      result_dict: dict
        A dictionary containing the loss for the current set of parameters, 
        the status of the current trial and a dictionary of parameters used
        for the trial
      """
      if not isinstance(space, dict):
        raise TypeError('Parameters must be provided as a dictionary')

      if space["grow_policy"]["grow_policy"] == "lossguide":
            max_leaves = space["grow_policy"].get("max_leaves")
            space["max_leaves"] = int(max_leaves)

      if space["booster"]["booster"] == "dart":
            sample_type = space["booster"].get("sample_type")
            normalize_type = space["booster"].get("normalize_type")
            rate_drop = space["booster"].get("rate_drop")
            skip_drop = space["booster"].get("skip_drop")
            space["sample_type"] = sample_type
            space["normalize_type"] = normalize_type
            space["rate_drop"] = rate_drop
            space["skip_drop"] = skip_drop

      space["grow_policy"] = space["grow_policy"]["grow_policy"]
      space["booster"] = space["booster"]["booster"] 

      print("Training with params: ")
      print(space)

      results = self.cross_validation(space)

      self.hyperopt_results = self.hyperopt_results.append({'params': results['params'], 'loss': results['loss'], 'variance': results['variance'], 'time': results['time']}, ignore_index = True)
      
      result_dict = {'loss': results['loss'], 'status': STATUS_OK, 'parameters': results['params']}

      return (result_dict)


    trials = Trials()
    optimize = fmin(fn = hyperopt_objective, 
                space = space, 
                algo = tpe.suggest, 
                trials = trials, 
                max_evals = self.max_evals,
                rstate = np.random.RandomState(seed = self.seed))
    
    best = self.hyperopt_results[['params', 'loss']].sort_values(by = 'loss', ascending = True).loc[0].to_dict()
    best = best['params']
    self.best_hyperopt_params = best
    print("The best HyperOpt hyperparameters are: ")
    print(best)
    print("\n")
    return best

  def optuna_tuning(self):

    """
    Applies the Optuna algorithm to tune the hyperparameters of the XGBoost
    model and stores the results in a dataframe

    Returns
    ----------
    best: dict
      A dictionary containing the set of parameters that resulted in the best
      value of the loss function using the Optuna optimization algorithm.
    """
    print("Starting Optuna hyperparameter tuning...")
    self.optuna_results = pd.DataFrame(columns = ['params', 'loss', 'variance', 'time'])
    
    def optuna_objective(trial):

      """
      Defines the objective function to be optimized by the Optuna algorithm

      Parameters
      ----------
      trial: Optuna trial object
        An Optuna trial that iterates over a set of hyperparameters to 
        minimize the objective function

      Returns
      -------
      cv_score: float
        The value of the loss function cross-validated over n-folds for the
        current set of hyperparameters for the trial
      """

      space = {
          "silent": 1,
          "objective": "binary:logistic",
          "eval_metric": "logloss",
          "verbosity": 2,
          "booster":
                    trial.suggest_categorical("booster", ["gbtree", "dart"]),
          "reg_lambda":
                    trial.suggest_int("reg_lambda", 1, 2),
          "reg_alpha":
                    trial.suggest_int("reg_alpha", 0, 10),
          "n_estimators":
                    trial.suggest_int("n_estimators", 5, 10),
          "max_depth":
                    trial.suggest_int("max_depth", 1, 14),
          "eta":
                    trial.suggest_uniform("eta", 0.025, 0.5),
          "gamma":
                    trial.suggest_uniform("gamma", 0.5, 1.0),
          "grow_policy":
                    trial.suggest_categorical(
                                    "grow_policy", ["depthwise", "lossguide"]),
          "min_child_weight":
                    trial.suggest_uniform("min_child_weight", 1, 10),
          "subsample":
                    trial.suggest_uniform("subsample", 0.7, 1),
          "colsample_bytree":
                    trial.suggest_uniform("colsample_bytree", 0.7, 1)
                }

      if space["grow_policy"] == "lossguide":
          space["max_leaves"] = trial.suggest_int("max_leaves", 0, 10)

      if space["booster"] == "dart":
          space["sample_type"] = trial.suggest_categorical(
                                        "sample_type", ["uniform", "weighted"])
          space["normalize_type"] = trial.suggest_categorical(
                                        "normalize_type", ["tree", "forest"])
          space["rate_drop"] = trial.suggest_uniform("rate_drop", 0, 1)
          space["skip_drop"] = trial.suggest_uniform("skip_drop", 0, 1)

      results = self.cross_validation(space)

      self.optuna_results = self.optuna_results.append({'params': results['params'],
                                                        'loss': results['loss'],
                                                        'variance': results['variance'],
                                                        'time': results['time']}, ignore_index = True)

      return results['loss']

    study = optuna.create_study(direction = "minimize", sampler = optuna.samplers.TPESampler(seed = self.seed))
    optimize = study.optimize(optuna_objective, n_trials = self.max_evals)
    best = study.best_params

    self.best_optuna_params = best

    print("The best Optuna hyperparameters are: ")
    print(best)
    print("\n")
    return best

  def random_search_tuning(self):

    """
    Applies the Random Search algorithm to tune the hyperparameters of the 
    XGBoost model and stores the results in a dataframe

    Returns
    -------
    best: dict
      A dictionary containing the set of parameters that resulted in the best
      value of the loss function using the Random Search optimization algorithm.
    """
    random.seed(self.seed)
    print("Starting Random Search hyperparameter tuning...")
    self.random_search_results = pd.DataFrame(columns = ['params', 'loss', 'variance', 'time'])

    def random_params():

      """
      Defines the hyperparameter search space for the Random Search algorithm

      Returns
      -------
      params: dict
        A dictionary containing the search-space for the Hyperopt optimization 
        algorithm
      """

      params = {
          "silent": [1],
          "objective": ["binary:logistic"],
          "eval_metric": ["logloss"],
          "booster": ["gbtree", "dart"],
          "reg_lambda": np.arange(1, 2, 0.1),
          "reg_alpha": np.arange(0, 10, 1),
          "verbosity": [2],
          "n_estimators": np.arange(5, 10, 1),
          "max_depth": np.arange(1, 14),
          "eta": np.arange(0.025, 0.5, 0.025),
          "gamma": np.arange(0.5, 1.0, 0.05),
          "grow_policy": ["depthwise", "lossguide"],
          "min_child_weight": np.arange(1, 10, 1),
          "subsample": np.arange(0.7, 1, 0.05),
          "colsample_bytree": np.arange(0.7, 1, 0.05),
          "max_leaves": np.arange(0, 10, 1),
          "sample_type": ["uniform", "weighted"],
          "normalize_type": ["tree", "forest"],
          "rate_drop": np.linspace(0, 1),
          "skip_drop": np.linspace(0, 1)
          }
      return params

    space = random_params()

    def random_objective(space):

      """
      Defines the objective function to be optimized by the Random Search
      algorithm

      Parameters
      ----------
      space: dict
        A dictionary of parameters to use for each trial of the Random Search
        optimization algorithm

      """
      if not isinstance(space, dict):
        raise TypeError('Parameters must be provided as a dictionary')

      print("Training with params: ")
      print(space)

      results = self.cross_validation(space)

      self.random_search_results = self.random_search_results.append({'params': results['params'], 'loss': results['loss'], 'variance': results['variance'], 'time': results['time']}, ignore_index = True)

    for i in range(self.max_evals):

      param_sample = {key: random.sample(list(value), 1)[0] for key, value in space.items()}

      random_objective(param_sample)

    best = self.random_search_results[['params', 'loss']].sort_values(by = 'loss', ascending = True).loc[0].to_dict()
    self.best_random_search_params = best['params']
    print("The best Random Search hyperparameters are: ")
    print(best)
    print("\n")
    return best

  def train_models(self, hyperopt = True, optuna = True, random_search = True):
    """
    Train the three best models obtained by the HyperOpt, Optuna and Random
    Search optimization algorithms

    Parameters
    ----------
    hyperopt: bool, default = True
      Whether to implement HyperOpt algorithm or not
    optuna: bool, default = True
      Whether to implement Optuna algorithm or not
    random_search = True
      Whether to implement the Random Search algorithm or not
    """
    random.seed(self.seed)
    print('Starting training...')
    self.split_data()
    self.output = {}

    if hyperopt:
      self.hyperopt_tuning()
      self.output['hyperopt'] = {}
      self.output['hyperopt']['params'] = self.best_hyperopt_params
      self.output['hyperopt']['model'] = self.best_hyperopt_model
    if optuna:
      self.optuna_tuning()
      self.output['optuna'] = {}
      self.output['optuna']['params'] = self.best_optuna_params
      self.output['optuna']['model'] = self.best_optuna_model
    if random_search:
      self.random_search_tuning()
      self.output['random_search'] = {}
      self.output['random_search']['params'] = self.best_random_search_params
      self.output['random_search']['model'] = self.best_random_search_model

    for i in self.output:
      model = xgb.train(self.output[i]['params'], self.dtrain)
      self.output[i]['model'] = model
    
    self.trained = True

  def test_models(self):
    """
    Evaluate the three best models obtained by the HyperOpt, Optuna and Random
    Search optimization algorithms on the testing set
    """
    if self.trained == False:
      raise Exception('Please train the models using the train_models method before testing')

    print('Starting Testing...')
    for i in self.output:
      prediction = self.output[i]['model'].predict(self.dtest)
      prediction = np.float64(prediction)
      self.output[i]['prediction'] = prediction
      score = log_loss(self.y_test, prediction)
      self.output[i]['score'] = score
    
    self.tested = True
    print(self.output)
    print('\n')

  def feature_importance(self, importance_type = 'gain'):

    """
    Calculates and plots feature importances of the models obtained by the
    HyperOpt, Optuna and Random Search optimization algorithms

    Parameters
    ----------
    importance_type: string, default = 'gain'
      Metric to evaluate feature importance
    """
    if importance_type not in ['weight', 'gain', 'cover', 'total_gain', 'total_cover', 'shap']:
      raise ValueError('Importance Type not supported. Must be among: \'weight', 'gain', 'cover', 'total_gain', 'total_cover', 'shap')
    if self.trained == False:
      raise Exception('Please train the models using the train_models method before calculating feature importances')
    
    if importance_type == 'shap':
      for i in self.output:
        if self.output[i]['params']['booster'] == 'gbtree':
          explainer = shap.TreeExplainer(self.output[i]['model'])
          importance = explainer.shap_values(self.x_train.values)
          fig = plt.figure()
          fig.suptitle(i + ' - Feature Importance - ' + importance_type)
          shap.summary_plot(importance, self.x_train.values)
    else:
      for i in self.output:
        importance = self.output[i]['model'].get_score(importance_type = importance_type)
        importance = {key:np.round(value, 2) for key, value in importance.items()}
        self.output[i]['feature_importance'] = importance
        xgb.plot_importance(importance, importance_type = importance_type)
        plt.title(i + ' - Feature Importance - ' + importance_type)
      

  def confusion_matrix(self):
    """
    Plots the confusion matrix for the models obtained by the HyperOpt, Optuna,
    and Random Search optimization algorithms
    """
    if self.tested == False:
      raise Exception('Please test the models using the test_models method before generating confusion matrix')

    for i in self.output:
      cm = sklearn.metrics.confusion_matrix(self.y_test, np.round(self.output[i]['prediction']))
      plt.figure()
      plt.title(i + ' - Confusion Matrix')
      sns.heatmap(cm, annot = True, fmt = 'd')

  def classification_report(self):
    """
    Prints the classification report for the predictions of the models obtained
    by the HyperOpt, Optuna, and Random Search optimization algorithms
    """
    if self.tested == False:
      raise Exception('Please test the models using the test_models method before generation classification report')

    for i in self.output:
      print(i)
      print(sklearn.metrics.classification_report(self.y_test, np.round(self.output[i]['prediction'])))

  def roc_curve(self):

    """
    Plots the roc_curve for the predictions of the models obtained
    by the HyperOpt, Optuna, and Random Search optimization algorithms
    """
    if self.tested == False:
      raise Exception('Please test the models using the test_models method before generating ROC curve')

    for i in self.output:
      lb = sklearn.preprocessing.LabelBinarizer()
      binarized_target = lb.fit_transform(self.y_test)
      fpr, tpr, thresholds = sklearn.metrics.roc_curve(binarized_target, self.output[i]['prediction'])
      roc_auc = sklearn.metrics.auc(fpr, tpr)

      plt.figure()
      plt.plot(fpr, tpr, lw = 2, label = 'ROC curve (area = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], lw = 2, linestyle = '--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(i + ' - ROC curve')
      plt.legend(loc="lower right")
      plt.show()

  def pr_curve(self):

    """
    Plots the pr_curve for the predictions of the models obtained
    by the HyperOpt, Optuna, and Random Search optimization algorithms
    """
    if self.tested == False:
          raise Exception('Please test the models using the test_models method before generating PR Curve')

    for i in self.output:
      average_precision = sklearn.metrics.average_precision_score(self.y_test, self.output[i]['prediction'])
      precision, recall, thresholds = sklearn.metrics.precision_recall_curve(self.y_test, self.output[i]['prediction'])
      fig = plt.figure()
      plt.title(i + ' - Precision-Recall Curve: AP={0:.2f}'.format(average_precision))
      plt.plot(recall, precision)
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      