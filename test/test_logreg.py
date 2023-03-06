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
import regression
from sklearn.preprocessing import StandardScaler
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

def test_prediction():
	# generate predictions and make sure they are close to the actual data
	import_data()
	# check that we get the expected number of predictions
	assert np.shape(preds) == (10,), "Incorrect number of predictions"
	# check our predictions
	assert true_preds == np.array([0.84940953, 0.51964634, 0.11676285, 0.83266918, 0.20462666,
       0.97990348, 0.30077071, 0.9744497 , 0.32878418, 0.57526369])
	assert np.allclose(preds, preds_check), "Predictions do not match expected"

def test_loss_function():
	import_data()
	# compare loss to sklearn loss
	my_losses = log_model.loss_function(y_val, log_model.make_prediction(X_val))
	sklearn_losses = log_loss(y_val, preds)
	assert np.is_close(my_losses, sklearn_losses), "Implemented loss does not match sklearn calculated loss"

def test_gradient():
	import_data()
	# calculate gradient
	gradient = log_model.calculate_gradient(y_val, X_val)
	assert gradient == np.array([-0.2784112, -0.1937338,  0.01551079,  0.,  0.,
        0.06515538,  0.07522936])

def test_training():
	import_data()
	# check that the trained model has updated weights
	# np.random.randn(num_feats + 1).flatten() is how the weights were initialized
	assert np.alltrue(log_model.W != np.random.randn(num_feats=6 + 1).flatten()), "Weights have not been updated after training"
