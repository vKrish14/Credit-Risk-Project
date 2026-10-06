"""
This module contains all the functions for making predictions with an ML model.
"""
import logging
import os
import sys
import pandas as pd

from sklearn.metrics import classification_report

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # root dir of project

# Add the parent directory to the system path (can import from anywhere)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from Prediction_Model import config, data_handling

fe_pipe = data_handling.load_pipeline('fe_pipeline_fitted')
model = data_handling.load_pipeline('XBG_model')
target_pipeline_fitted = data_handling.load_pipeline('target_pipeline_fitted')

def generate_prediction():
    """
    Load the test data, make predictions with the trained model, print the classification report,
    and return the predictions.
    
    Returns
    -------
    y_pred : array-like
        The predictions of the test data.
    """
    logging.info('Starting prediction')
    test_data = data_handling.load_data_and_sanitize('test_data.csv')
    y = test_data[config.TARGET]
    X = test_data.drop(config.TARGET,errors='ignore')
    y_pred_transformed = model.predict(fe_pipe.transform(X)) # do FE and then predict
    logging.info('Finished prediction')
    print(y_pred_transformed)
    y_pred = pd.Series(y_pred_transformed).map({0:'non defaulter',1:'defaulter'})
    print(classification_report(y, y_pred))
    return y_pred

if __name__ == '__main__':
    generate_prediction()