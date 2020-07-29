<<<<<<< HEAD
# ML-optimizations
Optimizing ML Projects
# Project Introduction:

The project aims to create and compare optimized ML models using Catboost, LightGBM and XGBoost on multiple data sources for binary classification problem

# Project Outline:

In this project, dataset is analyzed in several steps:
1. Data cleaning and preprocessing using Preprocess module.
2. Feature Engineering by creating new features using Featuretools package and feature selection methods. This methodology is done using Features module.
3. Fitting XGBoost, Catboost, and LightGBM models, and using Hyperopt and Optuna optimizers. 
4. Unit testing and integration testing of all the developed libraries using pytest.
5. Comparing of different fitted models and optimization methods against each other and AutoML of GCP.
6. Comparing the results with previous papers.

![mainfile+ETL](https://user-images.githubusercontent.com/56703496/85181382-c8da4980-b253-11ea-8bb4-2e30da00cb7b.png)

# Data Source:
The data used are available in public domain: 
1. Higgs dataset has been produced using Monte Carlo simulations at Physics & Astronomy, Univ. of California Irvine. The dataset can be found at (http://archive.ics.uci.edu/ml/datasets/HIGGS).

   It is a classification problem and identifies exotic particles in high-energy physics based on the sensors information(Signal process produces Higgs bosons (label 1) and a background process does not (label 0)).

   The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features which are high-level features derived by physicists to help discriminate between the two classes. For this project, we ignore the last 7 columns and use Featuretools python library (https://www.featuretools.com/) to create new features and compare with previous studies.
2. Credit card fraud dataset is an imbalanced classification problem ~ 500:1. The dataset is taken from Kaggle has 30 features where 28 are derived from PCA.
   This problem involves creating the custom evaluation function in LightGBM and XGBoost

# Contributors:

This is a group project and following are the list of contibutors:

[Ahmed Al-Baz](https://github.com/albazahm)

[Birkamel Kaur](https://github.com/Birkamal)

[Kshitij Mamgain](https://github.com/kshitijmamgain)

[Sasha Hajy Hassani](https://github.com/SHH116)

[Taraneh Kordi](https://github.com/Taraneh-K)

Supervisor: [Tanaby Zibamanzar Mofrad](https://github.com/tanabymofrad)

=======
# Fraud-Detection
Credit Card Fraud Detection
>>>>>>> origin
