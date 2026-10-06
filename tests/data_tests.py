import sys
import os

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # root dir of project

# Add the parent directory to the system path (can import from anywhere)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from Prediction_Model.data_handling import load_data_and_sanitize
from Prediction_Model import config

# Test Data import
df  = load_data_and_sanitize(config.FILE_NAME)

def test_data_import():
    """
    This test checks that the data has the expected number of columns.
    """
    assert df.shape[1] ==  27 , f"Expected 27 columns, got {df.shape[1]}"

# Range Constrains    
def test_data_range_loan_amnt():
    """
    Test that the range of the loan_amnt is within the expected values
    """
    assert df['loan_amnt'].min() >= 1
    assert df['loan_amnt'].max() <= 140000
    
def test_data_range_int_rate():
    """
    Test that the range of the int_rate is within the expected values
    """
    assert df['int_rate'].min() >= 2
    assert df['int_rate'].max() <= 40
    
# Uniqueness Constrains
def test_data_uniqueness():
    """
    Test that the data is unique
    """
    assert len(df) == len(df.drop_duplicates())
