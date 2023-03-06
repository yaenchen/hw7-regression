"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import (logreg, utils, )
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np
# (you will probably need to import more things here)

def import_data():
	# import the test data needed for all the following functions
	# load in the data, taken from main.py
	# Load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'Computed tomography of chest and abdomen',
			'Plain chest X-ray (procedure)',
			'Low Density Lipoprotein Cholesterol',
			'Creatinine',
			'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)

	# Scale the data, since values vary across feature. Note that we
	# fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	# For testing purposes, once you've added your code.
	# CAUTION: hyperparameters have not been optimized.
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)

	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	preds = log_model.make_prediction(X_val)[0:10]
	
	return (log_model, X_train, X_val, y_train, y_val, preds)

def test_prediction():
	# generate predictions and make sure they are close to the actual data
	log_model, X_train, X_val, y_train, y_val, preds = import_data()
	# check that we get the expected number of predictions
	assert np.shape(preds) == (10,), "Incorrect number of predictions"
	# check our predictions are between 0 and 1
	assert np.max(preds) < 1, "Predictions do not match expected"

def test_loss_function():
	log_model, X_train, X_val, y_train, y_val, preds = import_data()
	# compare loss to sklearn loss
	my_losses = log_model.loss_function(y_val, log_model.make_prediction(X_val))
	sklearn_losses = log_loss(y_val, log_model.make_prediction(X_val))
	assert np.isclose(my_losses, sklearn_losses), "Implemented loss does not match sklearn calculated loss"

def test_gradient():
	log_model, X_train, X_val, y_train, y_val, preds = import_data()
	# calculate gradient
	gradient = log_model.calculate_gradient(y_val, X_val)
	# check the a known gradient value, which is 0
	assert np.isclose(gradient[3], 0), "Gradient is not as expected"

def test_training():
	log_model, X_train, X_val, y_train, y_val, preds = import_data()
	# check that the trained model has updated weights
	# np.random.randn(num_feats + 1).flatten() is how the weights were initialized
	assert np.alltrue(log_model.W != np.random.randn(6 + 1).flatten()), "Weights have not been updated after training"
