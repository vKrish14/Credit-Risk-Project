"""
This module contains all the Pipelines for feature engineering.
"""
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif
    
import Prediction_Model.config as config



def get_age_of_credit(df):
    """
    Construct age of credit feature from earliest credit line and issue date.
    """
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    # missing values imputation for dates
    df['earliest_cr_line'] = df['earliest_cr_line'].ffill()
    df['issue_d'] = df['issue_d'].ffill()
    df['age_of_credit']= df['issue_d'].dt.year - df['earliest_cr_line'].dt.year
    return df[['age_of_credit']]

# Custom Transformer for feature extraction
class Distance_to_cluster(BaseEstimator, TransformerMixin):
    def __init__(self,n_clusters=10, gamma=0.8):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = config.RANDOM_SEED
    def fit(self,X,y=None,sample_weight=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X, sample_weight=sample_weight)
        return self
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Numerical pipelines
numerical_skewed_pipeline = Pipeline([
    ('select_numerical_skewed_features', FunctionTransformer(lambda X: X[config.NUM_SKEWED_FEATURES2])),
    ('FE_improvement_impute', SimpleImputer(strategy='median'))
])

numerical_features_pipeline = Pipeline([
    ('select_numerical_features', FunctionTransformer(lambda X: X[config.NUM_FEATURES])),
    ('FE_improvement_impute', SimpleImputer(strategy='mean')),
])

similarity_pipeline = Pipeline([
    ('select_location', FunctionTransformer(lambda X: X[['lat','lng']])),
    ('FE_construction_distance_to_cluster', Distance_to_cluster()),
])

all_num_features_pipeline = Pipeline([
    ('all_numerical',FeatureUnion([
        ('numerical_skewed_pipeline', numerical_skewed_pipeline),
        ('numerical_features_pipeline', numerical_features_pipeline),
        ('FE_construction_age_of_credit', FunctionTransformer(get_age_of_credit)),
        ('FE_construction_similarity', similarity_pipeline)        
    ])),
    # ('FE_construction_binning', KBinsDiscretizer(encode='ordinal')),
    ('FE_improvement_scaling',MinMaxScaler())
])

# Categorical pipelines
categorical_ordinal_pipeline_str = Pipeline([
    ('select_categorical_ordinal_features', FunctionTransformer(lambda X: X[config.CAT_ORDINAL_FEATURES_STR])),
    ('FE_improvement_impute', SimpleImputer(strategy='most_frequent')),
    ('FE_construction_ODE', OrdinalEncoder(categories=config.ORDER_MATRIX)),
    ('FE_improvement_scaling',MinMaxScaler())
])


categorical_ordinal_pipeline_int = Pipeline([
    ('select_categorical_ordinal_features', FunctionTransformer(lambda X: X[config.CAT_ORDINAL_FEATURES_INT])),
    ('FE_improvement_impute', SimpleImputer(strategy='most_frequent')),
    ('FE_construction_ODE', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)),
    ('FE_improvement_scaling',MinMaxScaler())
])

categorical_nominal_pipeline = Pipeline([
    ('select_categorical_nominal_features', FunctionTransformer(lambda X: X[config.CAT_NOMINAL_FEATURES2].apply(lambda x: x.astype('str').str.strip().str.lower()))), # select and clean nominal categorical features
    ('FE_improvement_impute', SimpleImputer(strategy='most_frequent')),
    ('FE_construction_OHE', OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=0.001,sparse_output=False))
])

# Combine all pipelines Numerical + Categorical
selected_FE = FeatureUnion([
        ('numerical_combined_pipeline',all_num_features_pipeline),
        ('categorical_ordinal_pipeline_str', categorical_ordinal_pipeline_str),
        ('categorical_ordinal_pipeline_int', categorical_ordinal_pipeline_int),
        ('categorical_nominal_pipeline', categorical_nominal_pipeline)
    ])

# Target pipeline for target variable encoding
# target_pipeline = LabelEncoder() # Not used because labelencoding encodes in lexicographical order thus making difficult for TunedThresholdClassifierCV tuning (need class 1 as 'Charged Off')

target_pipeline = Pipeline([
    ('target_ohe',FunctionTransformer(lambda x : x.map({'non defaulter':0,'defaulter':1}),
                                      inverse_func=lambda x : pd.Series(x).map({0:'non defaulter',1:'defaulter'}),
                                      check_inverse=False))
])

# Final pipeline
selected_FE_with_FS = Pipeline([
    ('feature_engineering_pipeline', selected_FE),
    ('feature_selection_pipeline',SelectKBest(score_func=mutual_info_classif))
])