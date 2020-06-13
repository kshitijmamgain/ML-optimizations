# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import pickle
import pandas as pd
import numpy as np
import random



def load_data(path, sample_rate=None):
	"""
	Loads the data from path and returns a dataframe on full / sample of data

	Args
	path: <string> path to the rawfile
	sample_rate: <None> or <float> pct sampling rate

	Returns:
	df <dataframe> data_read
	"""
	df = pd.read_csv(path)
	if sample_rate!=None:
		df = df.sample(frac=sample_rate)
	return df

def create_xy(df, target):
	"""
	Create Test and train set

	Args
	df <dataframe>: df dataframe
	target <int> or <string>: column index or label corresponding to target feature

	returns
	X <dataframe>: predictor features
	y <dataframe>: target_feature

	"""
	if isinstance(target, str):
		X = df.drop(columns=target)
		y = df.loc[:, target].values
	elif isinstance(target, int):
		X = df.drop(columns=df.columns[target])
		y = df.iloc[:, target].values

	return X, y











