"""
It includes functions for training an XGBoost model with feature engineering and saving the pipeline and test data.
It also includes functions for tuning the model's threshold adjustment.
"""
import logging
import os
import sys

from xgboost import XGBClassifier

# from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import mlflow
import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # root dir of project

# Add the parent directory to the system path (can import from anywhere)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Prediction_Model import config, FE_pipeline, data_handling,evaluation

SCORING = 'f1'

# Load and split data
df = data_handling.load_data_and_sanitize(config.FILE_NAME)
X = df.drop(config.TARGET, axis=1)
y = df[config.TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=config.RANDOM_SEED, stratify= y)
X_train[config.TARGET] = y_train
data_handling.save_data(X_train, 'train_data.csv')
X_test[config.TARGET] = y_test
data_handling.save_data(X_test, 'test_data.csv')
# Transform the target
y_train_transformed = FE_pipeline.target_pipeline.fit_transform(y_train)
# Transform test to match the target pipeline
y_test_transformed = FE_pipeline.target_pipeline.transform(y_test)
logging.info('Split data into train and test. Then, saved test data to test_data.csv')


def objective(trial):
    """
    Objective function for optuna tuning.

    This function takes a trial object from optuna and trains an ExtraTreesClassifier model with feature engineering.
    It logs the hyperparameters and the F1 score for the minority class in MLflow.
    """
    logging.info('Starting objective function for optuna trial')
    # Start an MLflow run
    with mlflow.start_run(nested=True,run_name=f"trial_{trial.number+1}"):
        # Define the hyperparameters to tune
        params = {
            # feature Engg params
            'fe_pipeline__feature_selection_pipeline__k':trial.suggest_int('k', 15, 30),
            'fe_pipeline__feature_engineering_pipeline__categorical_nominal_pipeline__FE_construction_OHE__min_frequency':trial.suggest_float('min_frequency', 0.001, 0.2),
            'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__n_clusters':trial.suggest_int('n_clusters', 7, 25),
            'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__gamma':trial.suggest_float('gamma', 0.1, 1),
            # 'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__FE_construction_binning__strategy':trial.suggest_categorical('strategy', ['uniform', 'quantile', 'kmeans']),
            # 'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__FE_construction_binning__n_bins':trial.suggest_int('n_bins', 3, 10),
            # model params
        #     'base_model__n_estimators': trial.suggest_int('n_estimators', 150, 500),
        #     'base_model__criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        #     'base_model__max_depth': trial.suggest_int('max_depth', 2, 32),
        #     'base_model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        #     'base_model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
        #     'base_model__max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        #     'base_model__bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        #     'base_model__random_state': config.RANDOM_SEED
        }
        # Define the pipeline with Feature Engineering
        model_with_fe = Pipeline([
            ('fe_pipeline', FE_pipeline.selected_FE_with_FS),
            ('base_model', XGBClassifier()) # this evaluates the FE pipeline
        ])
        # Set the hyperparameters (both model and FE pipeline)
        model_with_fe.set_params(**params)
        # Log hyperparameters in MLflow
        mlflow.log_params(params)
        # Train the model
        model_with_fe.fit(X_train, y_train_transformed)
        # Predict on the test set
        y_pred = model_with_fe.predict(X_test)
        # Calculate the metrics for class 1 (minority class)
        # f1_class_1 = f1_score(y_test_transformed, y_pred, pos_label=1)
        report  = classification_report(y_test_transformed, y_pred,output_dict=True)['1'] #  take output as dict so that we can log later in MLflow
        recall, f1= report['recall'], report['f1-score']
        # Log the F1 score in MLflow
        mlflow.log_metric('f1_score', round(f1,2))
        logging.info('Finished objective function for optuna trial')
    return round(f1,2)

def perform_feature_engineering(n_trials=50):
    """
    Perform feature engineering optimization using Optuna.
    
    This function will train a model with the best hyperparameters found by Optuna for feature engineering.
    It will also log the best hyperparameters and the best model in MLflow.
    Additionally, it will save the best model and the best feature engineering pipeline to the 'trained_models' folder.
    """
    logging.info('Starting training')
    mlflow.set_experiment("Feature Engineering Optuna Optimization")
    with mlflow.start_run(run_name="Final Optuna Optimized Model"):
        # Set MLflow tags for final model
        mlflow.set_tag("FE_engg", "EXTRA")
        mlflow.set_tag("objective", "maximize_f1_class_1")
        # Create an Optuna study to maximize the F1 score for class 1
        study = optuna.create_study(direction='maximize') # maximize f1 
        # Run the optimization with Optuna and log each trial in MLflow
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        # Get the best hyperparameters
        # best_trial = min(study.best_trials, key=lambda t: t.values[1]) # select the one with least k value assuming f1 is already maximized
        best_params = study.best_trial.params
        # Log the best trial
        logging.info("Best trial: %s",best_params)
        # Define the pipeline with Feature Engineering
        eval_model = Pipeline([
            ('fe_pipeline', FE_pipeline.selected_FE_with_FS),
            ('base_model', XGBClassifier()) # this evaluates the FE pipeline
        ])
        # recreate params from best trial
        params = {
            # feature Engg params
            'fe_pipeline__feature_selection_pipeline__k': best_params['k'],
            'fe_pipeline__feature_engineering_pipeline__categorical_nominal_pipeline__FE_construction_OHE__min_frequency': best_params['min_frequency'],
            'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__n_clusters': best_params['n_clusters'],
            'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__gamma': best_params['gamma'],
            # 'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__FE_construction_binning__strategy': best_params['strategy'],
            # 'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__FE_construction_binning__n_bins': best_params['n_bins'],
            # model params
            # 'base_model__n_estimators': best_params['n_estimators'],
            # 'base_model__criterion': best_params['criterion'],
            # 'base_model__max_depth': best_params['max_depth'],
            # 'base_model__min_samples_split': best_params['min_samples_split'],
            # 'base_model__min_samples_leaf': best_params['min_samples_leaf'],
            # 'base_model__max_features': best_params['max_features'],
            # 'base_model__bootstrap': best_params['bootstrap'],
        }
        # Set the hyperparameters (both model and FE pipeline)
        eval_model.set_params(**params)
        # Log the best hyperparameters in MLflow
        mlflow.log_params(params)
        # Train the model
        eval_model[:-1].set_output(transform='pandas')
        eval_model.fit(X_train, y_train_transformed)
        # Save features names (hack to save without using .get_feature_names_out() beacause FuncTransformer used)
        mlflow.log_text(str(eval_model[:-1].transform(X_train).columns), 'feature_names.txt')
        # Post tuning of selected best model (threshold adjustment as per business requirements)
        eval_model_tuned,report = evaluation.tune_model_threshold_adjustment(eval_model,
                                                X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                scoring=SCORING,
                                                target_pipeline=FE_pipeline.target_pipeline)
        mlflow.log_metrics(report['1']) # report is a nested dict for both class metrics => report['1'] gives all metrics for class 1 in dict format which will be logged
        mlflow.log_metric('threshold', eval_model_tuned.best_threshold_)
        mlflow.log_artifact("Prediction_Model/FE_pipeline.py")  # Log the pipeline as an artifact
        mlflow.log_artifact("Prediction_Model/config.py")  # Log the config file
        # mlflow.sklearn.log_model(eval_model_tuned, 'eval_model')
        # mlflow.sklearn.log_model(eval_model.named_steps['fe_pipeline'], 'fe_pipeline_fitted')

    data_handling.save_pipeline(eval_model, 'fe_eval_model')
    data_handling.save_pipeline(eval_model_tuned, 'fe_eval_tuned_model')
    data_handling.save_pipeline(eval_model.named_steps['fe_pipeline'], 'fe_pipeline_fitted') # Save best fe pipeline for actual model training in train.py
    data_handling.save_pipeline(FE_pipeline.target_pipeline, 'target_pipeline_fitted')
    logging.info('Model trained and saved pipeline to trained_models folder')
    # return   # TODO return SHARP with coefficients and report

if __name__ == '__main__':
    perform_feature_engineering(n_trials=100)
