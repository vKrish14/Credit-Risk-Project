"""
Test file for prediction
"""
import os
import sys
import numpy as np
import pytest

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # root dir of project

# Add the parent directory to the system path (can import from anywhere)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from Prediction_Model import config,data_handling

#Fixtures --> functions before test function --> ensure single_prediction
model = data_handling.load_pipeline('XBG_model')
fe_pipe = data_handling.load_pipeline('fe_pipeline_fitted')

@pytest.fixture
def single_prediction():
    """
    A fixture to return predictions on the test data.
    """
    test_data = data_handling.load_data_and_sanitize('test_data.csv')
    X,y = test_data.drop(config.TARGET,errors='ignore'),test_data[config.TARGET]
    pred = model.predict(fe_pipe.transform(X)) # do FE and then predict on test data (X)
    return pred

def test_single_pred_not_none(single_prediction): # output is not none
    """
    Test that the output of single_prediction is not None.
    """
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction): # data type is integer
    """
    Test that the data type of the first element in single_prediction is np.int64.
    """
    print(f"single_prediction[0]: {single_prediction[0]}, type: {type(single_prediction[0])}")
    assert isinstance(single_prediction[0],np.int64)