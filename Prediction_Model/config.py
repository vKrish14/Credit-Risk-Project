"""_
This module contains all the constants used in the project. 
It contains the following constants:

Naming
* FILE_NAME: The name of the data file
* URL: The URL of the data file if to be downloaded

Features
* CAT_ORDINAL_FEATURES : List of categorical features to be used in the ordinal encoder
* CAT_NOMINAL_FEATURES : List of categorical features to be used in the nominal encoder
* ORDER_MATRIX: Order matrix for ordinal encoder
* NUM_FEATURES: List of numerical features
* NUM_SKEWED_FEATURES: List of skewed numerical features
* TARGET: The name of the target variable
* POST_FE_FEATURES: List of features to be post-processed

Training constants
* N_JOBS: The number of jobs to run in parallel
* RANDOM_SEED: The random seed for reproducibility
"""
import os
# Naming
FILE_NAME = 'loan_reduced.csv'
URL = ''
PARENT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Features
CAT_ORDINAL_FEATURES_STR = ['term', 'grade','sub_grade','emp_length', 'verification_status']

CAT_ORDINAL_FEATURES_INT =['open_acc', 'pub_rec','total_acc','delinq_2yrs','inq_last_6mths','collections_12_mths_ex_med',
                           'open_acc_6m', 'open_il_6m', 'open_il_12m','open_il_24m', 'mths_since_rcnt_il', 'open_rv_12m', 'open_rv_24m',
                            'inq_fi', 'total_cu_tl', 'inq_last_12m']

term_order = [' 36 months', ' 60 months']
grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
sub_grade_order = [grade + str(i) for grade in grade_order for i in range(1,6)]
emp_length_order = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
verification_status_order = ['Not Verified', 'Verified', 'Source Verified']

ORDER_MATRIX = [term_order, grade_order, sub_grade_order, emp_length_order, verification_status_order]

# CAT_NOMINAL_FEATURES = ['home_ownership','purpose','title','initial_list_status','application_type'] # OHE with 1% threshold to be done
CAT_NOMINAL_FEATURES2 = ['home_ownership','purpose','title','emp_title','initial_list_status','application_type','zip_code','addr_state'] # OHE with 1% threshold to be done

NUM_FEATURES = ['loan_amnt','revol_util', 'last_pymnt_amnt','mths_since_last_delinq',
                'mths_since_last_major_derog', 'mths_since_last_record','total_bal_il', 'il_util',
                'max_bal_bc', 'all_util', 'total_rev_hi_lim','tot_coll_amt','tot_cur_bal']

# NUM_SKEWED_FEATURES = ['annual_inc','dti','open_acc', 'pub_rec','revol_bal', 'total_acc', 'mort_acc','pub_rec_bankruptcies']
NUM_SKEWED_FEATURES2 = ['annual_inc','dti','revol_bal']

POST_FE_FEATURES = ''

DEFAULT_VALUES = { # default values to be passed when feature is missing / not required from used because very less features are important
    
}
# TODO fill most frequent values here and then modify fastAPI, flask_app and streamlit_app accordingly with input params as input.update(dict) 

TARGET = 'loan_status'

# Training configs
N_JOBS = 8
RANDOM_SEED = 42