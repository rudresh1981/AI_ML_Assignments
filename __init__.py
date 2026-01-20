"""
Models Package
Contains all machine learning model implementations
"""

from . import logistic_regression
from . import decision_tree
from . import k_nn
from . import naive_bayes
from . import random_forest
from . import xgboost_model

# Model registry
MODEL_MODULES = {
    'Logistic Regression': logistic_regression,
    'Decision Tree': decision_tree,
    'k-NN': k_nn,
    'Naive Bayes': naive_bayes,
    'Random Forest': random_forest,
    'XGBoost': xgboost_model
}

__all__ = [
    'logistic_regression',
    'decision_tree',
    'k_nn',
    'naive_bayes',
    'random_forest',
    'xgboost_model',
    'MODEL_MODULES'
]
