import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools

from explore import EDA, countNull, tooSkewedFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
sns.set(style='ticks')
np.set_printoptions(precision=2)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

columns=dict({
    'class_label': tf.float32,
    'jet_1_b-tag': tf.float64,
    'jet_1_eta': tf.float64,
    'jet_1_phi': tf.float64,
    'jet_1_pt': tf.float64,
    'jet_2_b-tag': tf.float64,
    'jet_2_eta': tf.float64,
    'jet_2_phi': tf.float64,
    'jet_2_pt': tf.float64,
    'jet_3_b-tag': tf.float64,
    'jet_3_eta': tf.float64,
    'jet_3_phi': tf.float64,
    'jet_3_pt': tf.float64,
    'jet_4_b-tag': tf.float64,
    'jet_4_eta': tf.float64,
    'jet_4_phi': tf.float64,
    'jet_4_pt': tf.float64,
    'lepton_eta': tf.float64,
    'lepton_pT': tf.float64,
    'lepton_phi': tf.float64,
    'm_bb': tf.float64,
    'm_jj': tf.float64,
    'm_jjj': tf.float64,
    'm_jlv': tf.float64,
    'm_lv': tf.float64,
    'm_wbb': tf.float64,
    'm_wwbb': tf.float64,
    'missing_energy_magnitude': tf.float64,
    'missing_energy_phi': tf.float64,
})

# Read data into a pandas dataframe & assign column labels
raw_data = pd.read_csv('gs://higgs_data/HIGGS.csv.gz', compression = 'gzip', names = columns)

# Sample 1000 values
data_1000 = raw_data.sample(n=1000, random_state=1)

# Initial EDA
eda=EDA(target_label='class_label', is_target_cat=True)
eda.doEDA(data_1000)

# Replace missing values 

# Remove outliers

# Scale columns



