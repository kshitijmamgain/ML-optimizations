# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import random



def load_data(path, sample_rate):
	"""
	Loads the data from path and returns a dataframe on full / sample of data

	Args
	path <string> path to the rawfile
	sample_rate <float> pct sampling rate

	Returns:
	df <dataframe> data_read
	"""
	df = pd.read_csv(path)
	df = df.iloc[:, 1:-7]
	df = df.sample(frac=sample_rate)
	return df

def create_test_train(df_data):
	"""
	Create Test and train set

	Args
	df_data <dataframe>: df dataframe

	returns
	df_train <dataframe>: training set
	df_test <dataframe>: test set

	"""
	X = df_data.iloc[:, 1:]
	y = df_data.iloc[:, 0]
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

	return X_train, X_test, y_train, y_test

def preprocess_train_test(df, training_flag=True):

	"""
	Preprocess data for trainging/ testing

	Example:

			if training_flag is True, we are preprocessing the training set, else preprocessing the test set 


	Args:
	param1 df <dataframe>:
	...
	param training_flag <boolean> indicator of training/testing

	Returns:
	df <dataframe>

	"""
	
	return df



def preprocess_categorical_label_encoder(df,training_flag=True):
	"""Preprocess Categorical Data.

	Example: if trainging_flag -> True
	apply the encoding to the training set and pickle the 
	label encoder, other wise load and apply the encoder object. 

	Args:
	df < dataframe>: data
	path <string>: path to save the model

	Returns:
	df <dataframe>: transformed dataframe


	"""
	return df












