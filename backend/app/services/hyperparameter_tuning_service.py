import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb

class HyperparameterTuningService:
    """Service for hyperparameter tuning of ML models"""
    
    def __init__(self):
        self.classification_param_grids = {
            # Linear Models
            "LogisticRegression": {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            },
            "LogisticRegressionCV": {
                'Cs': [[0.01, 0.1, 1, 10]],
                'cv': [3, 5]
            },
            "RidgeClassifier": {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            "RidgeClassifierCV": {
                'alphas': [[0.1, 1.0, 10.0]]
            },
            "SGDClassifier": {
                'loss': ['hinge', 'log_loss'],
                'penalty': ['l2', 'l1'],
                'alpha': [0.0001, 0.001]
            },
            "PassiveAggressiveClassifier": {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000]
            },
            "Perceptron": {
                'penalty': ['l2', 'l1', None],
                'alpha': [0.0001, 0.001]
            },
            
            # SVM Models
            "SVC": {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            "NuSVC": {
                'nu': [0.25, 0.5, 0.75],
                'kernel': ['rbf', 'linear']
            },
            "LinearSVC": {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'max_iter': [2000]
            },
            
            # Tree Models
            "DecisionTreeClassifier": {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            "ExtraTreeClassifier": {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            "RandomForest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            },
            "ExtraTreesClassifier": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            },
            "GradientBoosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            "HistGradientBoosting": {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_iter': [100, 200],
                'max_depth': [5, 10, 20]
            },
            "AdaBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5]
            },
            "BaggingClassifier": {
                'n_estimators': [10, 50, 100],
                'max_samples': [0.5, 0.75, 1.0]
            },
            "XGBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            "LightGBM": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 70]
            },
            
            # Naive Bayes
            "GaussianNB": {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            },
            "BernoulliNB": {
                'alpha': [0.1, 0.5, 1.0]
            },
            
            # Discriminant Analysis
            "LinearDiscriminantAnalysis": {
                'solver': ['svd', 'lsqr']
            },
            "QuadraticDiscriminantAnalysis": {
                'reg_param': [0.0, 0.1, 0.5]
            },
            
            # Neighbors
            "KNN": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            "NearestCentroid": {
                'metric': ['euclidean', 'manhattan']
            },
            
            # Neural Network
            "MLPClassifier": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001]
            },
            
            # Semi-Supervised
            "LabelPropagation": {
                'kernel': ['rbf', 'knn'],
                'n_neighbors': [7, 10]
            },
            "LabelSpreading": {
                'kernel': ['rbf', 'knn'],
                'n_neighbors': [7, 10]
            },
            
            # Calibration
            "CalibratedClassifierCV": {
                'method': ['sigmoid', 'isotonic'],
                'cv': [3, 5]
            }
        }
        
        self.regression_param_grids = {
            # Linear Models
            "Linear Regression": {
                'fit_intercept': [True, False]
            },
            "Ridge": {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            "RidgeCV": {
                'alphas': [[0.1, 1.0, 10.0]]
            },
            "Lasso": {
                'alpha': [0.1, 1.0, 10.0]
            },
            "LassoCV": {
                'cv': [3, 5]
            },
            "LassoLars": {
                'alpha': [0.1, 1.0, 10.0]
            },
            "LassoLarsCV": {
                'cv': [3, 5]
            },
            "LassoLarsIC": {
                'criterion': ['aic', 'bic']
            },
            "Lars": {
                'n_nonzero_coefs': [50, 100, 200]
            },
            "LarsCV": {
                'cv': [3, 5]
            },
            "ElasticNet": {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            "ElasticNetCV": {
                'l1_ratio': [[0.1, 0.5, 0.9]],
                'cv': [3, 5]
            },
            "BayesianRidge": {
                'n_iter': [100, 300],
                'alpha_1': [1e-6, 1e-5],
                'lambda_1': [1e-6, 1e-5]
            },
            "HuberRegressor": {
                'epsilon': [1.35, 1.5, 2.0],
                'alpha': [0.0001, 0.001]
            },
            "SGDRegressor": {
                'loss': ['squared_error', 'huber'],
                'penalty': ['l2', 'l1'],
                'alpha': [0.0001, 0.001]
            },
            "PassiveAggressiveRegressor": {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000]
            },
            "OrthogonalMatchingPursuit": {
                'n_nonzero_coefs': [5, 10, 20]
            },
            "OrthogonalMatchingPursuitCV": {
                'cv': [3, 5]
            },
            "RANSACRegressor": {
                'min_samples': [0.5, 0.75],
                'max_trials': [100, 200]
            },
            "PoissonRegressor": {
                'alpha': [0.1, 1.0, 10.0]
            },
            "TweedieRegressor": {
                'power': [0, 1, 2],
                'alpha': [0.1, 1.0]
            },
            "GammaRegressor": {
                'alpha': [0.1, 1.0, 10.0]
            },
            
            # Tree Models
            "DecisionTreeRegressor": {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            "ExtraTreeRegressor": {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            "RandomForest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            },
            "ExtraTreesRegressor": {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            },
            "GradientBoosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            "HistGradientBoosting": {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_iter': [100, 200],
                'max_depth': [5, 10, 20]
            },
            "AdaBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5]
            },
            "BaggingRegressor": {
                'n_estimators': [10, 50, 100],
                'max_samples': [0.5, 0.75, 1.0]
            },
            "XGBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            "LightGBM": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 70]
            },
            
            # SVM Models
            "SVR": {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            "NuSVR": {
                'nu': [0.25, 0.5, 0.75],
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            },
            "LinearSVR": {
                'C': [0.1, 1, 10],
                'epsilon': [0.0, 0.1, 0.2]
            },
            
            # Other Models
            "KNN": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            "KernelRidge": {
                'alpha': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            },
            "GaussianProcess": {
                'alpha': [1e-10, 1e-5],
                'n_restarts_optimizer': [0, 5]
            },
            "MLPRegressor": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001]
            },
            "DummyRegressor": {
                'strategy': ['mean', 'median']
            },
            "TransformedTargetRegressor": {
                'transformer': [None]
            }
        }
    
    def tune_model(self, model, model_name, X_train, y_train, problem_type, use_randomized=True):
        """Tune a single model using GridSearchCV or RandomizedSearchCV"""
        
        param_grid = self.classification_param_grids.get(model_name) if problem_type == "classification" else self.regression_param_grids.get(model_name)
        
        if not param_grid:
            return None, None, None
        
        # Choose search method based on dataset size
        n_samples = len(X_train)
        
        if use_randomized and n_samples > 1000:
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=20,
                cv=5,
                scoring='accuracy' if problem_type == 'classification' else 'r2',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        else:
            search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='accuracy' if problem_type == 'classification' else 'r2',
                n_jobs=-1,
                verbose=0
            )
        
        try:
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_, search.best_score_
        except Exception as e:
            print(f"Error tuning {model_name}: {e}")
            return None, None, None
