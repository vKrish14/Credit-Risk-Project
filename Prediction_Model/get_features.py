"""
Feature Engineering and Model Evaluation

This module:
1. Loads and sanitizes the loan dataset.
2. Splits the data into training and testing sets.
3. Optimizes feature-engineering parameters using Optuna.
4. Trains an XGBoost model with the selected feature-engineering pipeline.
5. Tunes the classification threshold.
6. Saves the trained pipelines locally for use by train.py and Streamlit.

MLflow tracking has been removed from this local training workflow because
the current Python 3.13 environment has MLflow/skops serialization issues.
"""

import logging
import os
import sys

import optuna
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


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


# ============================================================
# LOAD AND SPLIT DATA
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Loading dataset...")

df = data_handling.load_data_and_sanitize(config.FILE_NAME)

X = df.drop(config.TARGET, axis=1)
y = df[config.TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=config.RANDOM_SEED,
    stratify=y
)


# Save training data
X_train_with_target = X_train.copy()
X_train_with_target[config.TARGET] = y_train

data_handling.save_data(
    X_train_with_target,
    'train_data.csv'
)


# Save testing data
X_test_with_target = X_test.copy()
X_test_with_target[config.TARGET] = y_test

data_handling.save_data(
    X_test_with_target,
    'test_data.csv'
)


# ============================================================
# TARGET TRANSFORMATION
# ============================================================

logging.info("Transforming target variable...")

y_train_transformed = (
    FE_pipeline.target_pipeline.fit_transform(y_train)
)

y_test_transformed = (
    FE_pipeline.target_pipeline.transform(y_test)
)

logging.info(
    "Split data into train and test sets."
)


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):
    """
    Optuna objective function.

    The function tunes the feature-engineering parameters and
    evaluates the resulting XGBoost model using F1 score for
    the minority class.
    """

    logging.info(
        "Starting Optuna trial %s",
        trial.number + 1
    )

    # --------------------------------------------------------
    # Feature-engineering parameters
    # --------------------------------------------------------

    params = {
        'fe_pipeline__feature_selection_pipeline__k':
            trial.suggest_int(
                'k',
                15,
                30
            ),

        'fe_pipeline__feature_engineering_pipeline__categorical_nominal_pipeline__FE_construction_OHE__min_frequency':
            trial.suggest_float(
                'min_frequency',
                0.001,
                0.2
            ),

        'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__n_clusters':
            trial.suggest_int(
                'n_clusters',
                7,
                25
            ),

        'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__gamma':
            trial.suggest_float(
                'gamma',
                0.1,
                1.0
            )
    }


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    model_with_fe = Pipeline([
        (
            'fe_pipeline',
            FE_pipeline.selected_FE_with_FS
        ),

        (
            'base_model',
            XGBClassifier()
        )
    ])


    # Apply Optuna parameters
    model_with_fe.set_params(**params)


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model_with_fe.fit(
        X_train,
        y_train_transformed
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model_with_fe.predict(
        X_test
    )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    report = classification_report(
        y_test_transformed,
        y_pred,
        output_dict=True
    )['1']


    f1 = report['f1-score']


    logging.info(
        "Trial %s completed. F1 score: %.2f",
        trial.number + 1,
        f1
    )


    return round(f1, 2)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def perform_feature_engineering(n_trials=50):
    """
    Perform feature-engineering optimization using Optuna.

    The best feature-engineering pipeline and target pipeline
    are saved to Prediction_Model/trained_models/.
    """

    logging.info(
        "Starting feature-engineering optimization..."
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
    # Best trial
    # --------------------------------------------------------

    best_trial = study.best_trial

    best_params = best_trial.params

    logging.info(
        "Best trial: %s",
        best_trial.number
    )

    logging.info(
        "Best F1 score: %.2f",
        study.best_value
    )

    logging.info(
        "Best parameters: %s",
        best_params
    )


    # ========================================================
    # CREATE FINAL EVALUATION MODEL
    # ========================================================

    eval_model = Pipeline([
        (
            'fe_pipeline',
            FE_pipeline.selected_FE_with_FS
        ),

        (
            'base_model',
            XGBClassifier()
        )
    ])


    # --------------------------------------------------------
    # Recreate parameters using best Optuna values
    # --------------------------------------------------------

    params = {
        'fe_pipeline__feature_selection_pipeline__k':
            best_params['k'],

        'fe_pipeline__feature_engineering_pipeline__categorical_nominal_pipeline__FE_construction_OHE__min_frequency':
            best_params['min_frequency'],

        'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__n_clusters':
            best_params['n_clusters'],

        'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__gamma':
            best_params['gamma']
    }


    # Apply best parameters
    eval_model.set_params(
        **params
    )


    # ========================================================
    # TRAIN FINAL FEATURE-ENGINEERING MODEL
    # ========================================================

    logging.info(
        "Training final feature-engineering model..."
    )


    # Keep transformed features in pandas format
    eval_model[:-1].set_output(
        transform='pandas'
    )


    eval_model.fit(
        X_train,
        y_train_transformed
    )


    # ========================================================
    # THRESHOLD TUNING
    # ========================================================

    logging.info(
        "Tuning classification threshold..."
    )


    eval_model_tuned, report = (
        evaluation.tune_model_threshold_adjustment(
            eval_model,
            X_train,
            y_train,
            X_test,
            y_test,
            scoring=SCORING,
            target_pipeline=FE_pipeline.target_pipeline
        )
    )


    logging.info(
        "Best threshold: %.2f",
        eval_model_tuned.best_threshold_
    )


    # ========================================================
    # SAVE PIPELINES
    # ========================================================

    logging.info(
        "Saving trained pipelines..."
    )


    # Complete evaluation model
    data_handling.save_pipeline(
        eval_model,
        'fe_eval_model'
    )


    # Threshold-tuned evaluation model
    data_handling.save_pipeline(
        eval_model_tuned,
        'fe_eval_tuned_model'
    )


    # Feature-engineering pipeline
    # This is used later by train.py
    data_handling.save_pipeline(
        eval_model.named_steps['fe_pipeline'],
        'fe_pipeline_fitted'
    )


    # Target transformation pipeline
    data_handling.save_pipeline(
        FE_pipeline.target_pipeline,
        'target_pipeline_fitted'
    )


    logging.info(
        "Feature-engineering pipelines saved successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    perform_feature_engineering(
        n_trials=100
    )