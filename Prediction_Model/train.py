"""
Final XGBoost Model Training

This module:
1. Loads the train/test datasets.
2. Loads the fitted feature-engineering pipeline.
3. Optimizes XGBoost hyperparameters using Optuna.
4. Trains the final XGBoost model.
5. Tunes the classification threshold.
6. Saves the final model and target pipeline locally.

MLflow tracking has been removed from this local training workflow.
"""

import logging
import os
import sys

import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


# ============================================================
# PROJECT PATH
# ============================================================

parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# ============================================================
# PROJECT MODULES
# ============================================================

from Prediction_Model import (
    config,
    FE_pipeline,
    data_handling,
    evaluation
)


# ============================================================
# SETTINGS
# ============================================================

SCORING = 'f1'


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================
# LOAD TRAINING AND TEST DATA
# ============================================================

logging.info("Loading training data...")

df_train = data_handling.load_data_and_sanitize(
    'train_data.csv'
)

X_train = df_train.drop(
    config.TARGET,
    axis=1
)

y_train = df_train[config.TARGET]


logging.info("Loading testing data...")

df_test = data_handling.load_data_and_sanitize(
    'test_data.csv'
)

X_test = df_test.drop(
    config.TARGET,
    axis=1
)

y_test = df_test[config.TARGET]


# ============================================================
# LOAD FITTED FEATURE-ENGINEERING PIPELINE
# ============================================================

logging.info(
    "Loading fitted feature-engineering pipeline..."
)

feature_engg = data_handling.load_pipeline(
    'fe_pipeline_fitted'
)


# ============================================================
# TRANSFORM FEATURES
# ============================================================

logging.info(
    "Transforming training features..."
)

X_train_transformed = feature_engg.transform(
    X_train
)


logging.info(
    "Transforming testing features..."
)

X_test_transformed = feature_engg.transform(
    X_test
)


# ============================================================
# TARGET TRANSFORMATION
# ============================================================

logging.info(
    "Transforming target variable..."
)

y_train_transformed = (
    FE_pipeline.target_pipeline.fit_transform(
        y_train
    )
)

y_test_transformed = (
    FE_pipeline.target_pipeline.transform(
        y_test
    )
)


logging.info(
    "Training and testing data prepared."
)


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):
    """
    Optuna objective function.

    Tunes XGBoost hyperparameters and maximizes
    F1 score for the minority class.
    """

    logging.info(
        "Starting XGBoost trial %s",
        trial.number + 1
    )


    # --------------------------------------------------------
    # XGBoost hyperparameters
    # --------------------------------------------------------

    params = {

        'max_depth':
            trial.suggest_int(
                'max_depth',
                2,
                10
            ),

        'learning_rate':
            trial.suggest_float(
                'learning_rate',
                0.01,
                0.3,
                log=True
            ),

        'n_estimators':
            trial.suggest_int(
                'n_estimators',
                100,
                1000
            ),

        'gamma':
            trial.suggest_float(
                'gamma',
                0.001,
                1.0,
                log=True
            ),

        'subsample':
            trial.suggest_float(
                'subsample',
                0.5,
                1.0
            ),

        'colsample_bytree':
            trial.suggest_float(
                'colsample_bytree',
                0.5,
                1.0
            ),

        'lambda':
            trial.suggest_float(
                'lambda',
                0.001,
                10.0,
                log=True
            ),

        'alpha':
            trial.suggest_float(
                'alpha',
                0.001,
                10.0,
                log=True
            ),

        'tree_method':
            'hist',

        'eval_metric':
            'aucpr',

        'scale_pos_weight':
            trial.suggest_float(
                'scale_pos_weight',
                1.0,
                10.0
            ),

        'min_child_weight':
            trial.suggest_int(
                'min_child_weight',
                1,
                10
            ),

        'grow_policy':
            trial.suggest_categorical(
                'grow_policy',
                [
                    'depthwise',
                    'lossguide'
                ]
            )
    }


    # --------------------------------------------------------
    # Create XGBoost model
    # --------------------------------------------------------

    xgb_model = XGBClassifier(
        **params
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    xgb_model.fit(
        X_train_transformed,
        y_train_transformed
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = xgb_model.predict(
        X_test_transformed
    )


    # --------------------------------------------------------
    # F1 score
    # --------------------------------------------------------

    f1_class_1 = f1_score(
        y_test_transformed,
        y_pred,
        pos_label=1
    )


    logging.info(
        "Trial %s completed. F1 = %.4f",
        trial.number + 1,
        f1_class_1
    )


    return f1_class_1


# ============================================================
# FINAL TRAINING
# ============================================================

def perform_training(
    n_trials=100
):
    """
    Optimize XGBoost hyperparameters, train the final model,
    tune its classification threshold, and save the resulting
    model locally.
    """

    logging.info(
        "Starting XGBoost hyperparameter optimization..."
    )


    # --------------------------------------------------------
    # Create Optuna study
    # --------------------------------------------------------

    study = optuna.create_study(
        direction='maximize'
    )


    # --------------------------------------------------------
    # Run optimization
    # --------------------------------------------------------

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )


    # --------------------------------------------------------
    # Best parameters
    # --------------------------------------------------------

    best_params = (
        study.best_trial.params
    )


    logging.info(
        "Best trial: %s",
        study.best_trial.number
    )

    logging.info(
        "Best F1 score: %.4f",
        study.best_value
    )

    logging.info(
        "Best parameters: %s",
        best_params
    )


    # ========================================================
    # TRAIN FINAL MODEL
    # ========================================================

    logging.info(
        "Training final XGBoost model..."
    )


    XGB_with_FE_best = XGBClassifier(
        **best_params
    )


    XBG_model = XGB_with_FE_best.fit(
        X_train_transformed,
        y_train_transformed
    )


    # ========================================================
    # THRESHOLD TUNING
    # ========================================================

    logging.info(
        "Starting threshold tuning..."
    )


    XBG_model_tuned, report = (
        evaluation.tune_model_threshold_adjustment(
            XBG_model,
            X_train_transformed,
            y_train,
            X_test_transformed,
            y_test,
            scoring=SCORING,
            target_pipeline=FE_pipeline.target_pipeline
        )
    )


    logging.info(
        "Best threshold: %.2f",
        XBG_model_tuned.best_threshold_
    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    logging.info(
        "Saving final XGBoost model..."
    )


    data_handling.save_pipeline(
        XBG_model_tuned,
        'XBG_model'
    )


    # Save target pipeline again
    data_handling.save_pipeline(
        FE_pipeline.target_pipeline,
        'target_pipeline_fitted'
    )


    logging.info(
        "Final XGBoost model saved successfully."
    )

    logging.info(
        "Model artifact: trained_models/XBG_model.pkl"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    perform_training(
        n_trials=100
    )