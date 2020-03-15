import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
raw_data=pd.read_csv(path, compression = 'gzip', names = columns)

# Identify outliers
def outlier_index(df, feature):
        values=sorted(df[feature])
        q1, q3= np.percentile(values,[25,75])
        iqr=q3-q1
        lower_bound = q1 -(1.5 * iqr) 
        upper_bound = q3 +(1.5 * iqr)

        i=0
        outliers=[]
        for y in df[feature]:
            if y<lower_bound and y>upper_bound:
                outliers.append(i)
            i=i+1
        return outliers
    
for col in raw_features:
    print(outlier_index(raw_features, col))

# Drop outliers
   
# Replace missing values 
imp=SimpleImputer(missing_values=-999.0, strategy='mean')
raw_features=pd.DataFrame(imp.fit_transform(raw_features), columns=[1:], columns=columns[1:])

stc=StandardScaler()
raw_features=pd.DataFrame(stc.fit_transform(raw_features), columns=columns[1:])


