import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve,
                             r2_score, mean_squared_error, mean_absolute_error)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, 
                                   RidgeCV, LassoCV, ElasticNetCV, LassoLars, LassoLarsCV, LassoLarsIC,
                                   Lars, LarsCV, BayesianRidge, HuberRegressor, SGDRegressor, 
                                   PassiveAggressiveRegressor, OrthogonalMatchingPursuit, 
                                   OrthogonalMatchingPursuitCV, RANSACRegressor, PoissonRegressor,
                                   TweedieRegressor, GammaRegressor, LogisticRegressionCV, SGDClassifier,
                                   RidgeClassifier, RidgeClassifierCV, PassiveAggressiveClassifier, Perceptron)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, 
                              GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor,
                              ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor,
                              ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
from ..utils.graphs import GraphGenerator
from ..utils.file_manager import FileManager
from .hyperparameter_tuning_service import HyperparameterTuningService

class ModelTrainingService:
    def __init__(self):
        self.tuning_service = HyperparameterTuningService()
    
    def generate_model_results_table(self, results: list, problem_type: str) -> str:
        """
        GEREKSİNİM 2A: LazyPredict benzeri model sonuçları tablosu
        Generate a comprehensive Markdown table with all model results.
        
        Args:
            results (list): List of model result dictionaries
            problem_type (str): 'classification' or 'regression'
        
        Returns:
            str: Markdown formatted results table
        """
        markdown = "## 🎯 Model Performans Özeti (Model Performance Summary)\n\n"
        
        # Separate successful and failed models
        successful_models = [r for r in results if r.get('status') == 'success']
        failed_models = [r for r in results if r.get('status') == 'failed']
        
        if problem_type == "classification":
            markdown += "| Model | Accuracy | F1 Score | Recall | Precision | Eğitim Süresi (s) | Status |\n"
            markdown += "|-------|----------|----------|--------|-----------|-------------------|--------|\n"
            
            # Sort by accuracy (descending) - successful models first
            sorted_results = sorted(successful_models, key=lambda x: x.get('accuracy', 0), reverse=True)
            
            for result in sorted_results:
                model_name = result.get('model_name', 'Unknown')
                accuracy = result.get('accuracy', 0.0)
                f1 = result.get('f1_score', 0.0)
                recall = result.get('recall', 0.0)
                precision = result.get('precision', 0.0)
                time_taken = result.get('training_time', 0.0)
                
                markdown += f"| {model_name} | {accuracy:.3f} | {f1:.3f} | {recall:.3f} | {precision:.3f} | {time_taken:.2f} | ✅ Success |\n"
            
            # Add failed models at the end
            for result in failed_models:
                model_name = result.get('model_name', 'Unknown')
                error_msg = result.get('error_message', 'Unknown error')[:50]  # Truncate long errors
                markdown += f"| {model_name} | N/A | N/A | N/A | N/A | N/A | ❌ Failed: {error_msg} |\n"
        
        else:  # regression
            markdown += "| Model | R² Score | RMSE | MAE | MSE | Eğitim Süresi (s) | Status |\n"
            markdown += "|-------|----------|------|-----|-----|-------------------|--------|\n"
            
            # Sort by R² (descending) - successful models first
            sorted_results = sorted(successful_models, key=lambda x: x.get('r2_score', 0), reverse=True)
            
            for result in sorted_results:
                model_name = result.get('model_name', 'Unknown')
                r2 = result.get('r2_score', 0.0)
                rmse = result.get('rmse', 0.0)
                mae = result.get('mae', 0.0)
                mse = result.get('mse', 0.0)
                time_taken = result.get('training_time', 0.0)
                
                markdown += f"| {model_name} | {r2:.3f} | {rmse:.3f} | {mae:.3f} | {mse:.3f} | {time_taken:.2f} | ✅ Success |\n"
            
            # Add failed models at the end
            for result in failed_models:
                model_name = result.get('model_name', 'Unknown')
                error_msg = result.get('error_message', 'Unknown error')[:50]  # Truncate long errors
                markdown += f"| {model_name} | N/A | N/A | N/A | N/A | N/A | ❌ Failed: {error_msg} |\n"
        
        # Add best model highlight (only from successful models)
        if sorted_results:
            best_model = sorted_results[0]['model_name']
            best_score = sorted_results[0].get('accuracy' if problem_type == 'classification' else 'r2_score', 0)
            metric_name = 'Accuracy' if problem_type == 'classification' else 'R² Score'
            markdown += f"\n**🏆 En İyi Model (Best Model):** {best_model} ({metric_name}: {best_score:.3f})\n"
        
        # Add failure summary if any
        if failed_models:
            markdown += f"\n**⚠️ Başarısız Modeller (Failed Models):** {len(failed_models)}/{len(results)}\n"
            markdown += "\nBazı modeller veri uyumsuzluğu nedeniyle eğitilemedi. Başarılı modeller yukarıda listelenmiştir.\n"
        
        return markdown
    def detect_problem_type(self, df: pd.DataFrame, target_column: str):
        target = df[target_column]
        if pd.api.types.is_numeric_dtype(target):
            # Check if it has few unique values (likely classification) or many (regression)
            unique_values = target.nunique()
            if unique_values < 20: # Heuristic threshold
                return "classification"
            else:
                return "regression"
        else:
            return "classification"

    def train_models(self, df: pd.DataFrame, target_column: str, with_tuning: bool = False):
        problem_type = self.detect_problem_type(df, target_column)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = []
        
        if problem_type == "classification":
            models = {
                # Linear Models
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "LogisticRegressionCV": LogisticRegressionCV(max_iter=1000),
                "RidgeClassifier": RidgeClassifier(),
                "RidgeClassifierCV": RidgeClassifierCV(),
                "SGDClassifier": SGDClassifier(),
                "PassiveAggressiveClassifier": PassiveAggressiveClassifier(),
                "Perceptron": Perceptron(),
                
                # SVM Models
                "SVC": SVC(probability=True),
                "NuSVC": NuSVC(probability=True),
                "LinearSVC": LinearSVC(max_iter=2000),
                
                # Tree Models
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "ExtraTreeClassifier": ExtraTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "ExtraTreesClassifier": ExtraTreesClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "HistGradientBoosting": HistGradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "BaggingClassifier": BaggingClassifier(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                
                # Naive Bayes
                "GaussianNB": GaussianNB(),
                "BernoulliNB": BernoulliNB(),
                
                # Discriminant Analysis
                "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
                "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
                
                # Neighbors
                "KNN": KNeighborsClassifier(),
                "NearestCentroid": NearestCentroid(),
                
                # Neural Network
                "MLPClassifier": MLPClassifier(max_iter=1000),
                
                # Semi-Supervised
                "LabelPropagation": LabelPropagation(),
                "LabelSpreading": LabelSpreading(),
                
                # Calibration
                "CalibratedClassifierCV": CalibratedClassifierCV()
            }
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = lgb.LGBMClassifier(verbose=-1)
            
            total_models = len(models)
            for idx, (name, model) in enumerate(models.items(), 1):
                print(f"STATUS: Training model {idx}/{total_models}: {name}...")
                try:
                    import time
                    start_time = time.time()
                    
                    base_score = None
                    tuned_score = None
                    best_params = None
                    
                    # Train baseline model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                    
                    training_time = time.time() - start_time
                    
                    # Baseline metrics
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted')
                    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None and len(np.unique(y_test)) == 2 else None
                    
                    base_score = acc
                    
                    # Hyperparameter tuning if requested
                    if with_tuning:
                        tuning_start = time.time()
                        tuned_model, best_params, tuned_score = self.tuning_service.tune_model(
                            model, name, X_train, y_train, problem_type
                        )
                        if tuned_model:
                            model = tuned_model
                            y_pred = model.predict(X_test)
                            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                            
                            # Recalculate metrics with tuned model
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            precision = precision_score(y_test, y_pred, average='weighted')
                            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None and len(np.unique(y_test)) == 2 else None
                            
                            training_time += time.time() - tuning_start
                    
                    # GEREKSİNİM 2: Grafikler kaldırıldı, sadece metrikler saklanıyor
                    # Save model
                    model_path = FileManager.save_model(model, f"{name.replace(' ', '_')}.joblib")
                    
                    results.append({
                        "model_name": name,
                        "accuracy": acc,
                        "f1_score": f1,
                        "recall": recall,
                        "precision": precision,
                        "roc_auc": roc_auc,
                        "training_time": training_time,
                        "base_score": base_score,
                        "tuned_score": tuned_score,
                        "improvement": (tuned_score - base_score) if (tuned_score and base_score) else None,
                        "best_params": best_params,
                        "model_path": model_path,
                        "status": "success"
                    })
                    print(f"✅ {name} trained successfully (Accuracy: {acc:.3f})")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ FAILED: {name} - {error_msg}")
                    print(f"⏭️  Skipping {name} and continuing with next model...")
                    
                    # Add failed model to results with error status
                    results.append({
                        "model_name": name,
                        "accuracy": 0.0,
                        "f1_score": 0.0,
                        "recall": 0.0,
                        "precision": 0.0,
                        "roc_auc": None,
                        "training_time": 0.0,
                        "base_score": None,
                        "tuned_score": None,
                        "improvement": None,
                        "best_params": None,
                        "model_path": None,
                        "status": "failed",
                        "error_message": error_msg
                    })
                    
        else: # Regression
            models = {
                # Linear Models
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "RidgeCV": RidgeCV(),
                "Lasso": Lasso(),
                "LassoCV": LassoCV(),
                "LassoLars": LassoLars(),
                "LassoLarsCV": LassoLarsCV(),
                "LassoLarsIC": LassoLarsIC(),
                "Lars": Lars(),
                "LarsCV": LarsCV(),
                "ElasticNet": ElasticNet(),
                "ElasticNetCV": ElasticNetCV(),
                "BayesianRidge": BayesianRidge(),
                "HuberRegressor": HuberRegressor(),
                "SGDRegressor": SGDRegressor(),
                "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
                "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
                "OrthogonalMatchingPursuitCV": OrthogonalMatchingPursuitCV(),
                "RANSACRegressor": RANSACRegressor(),
                "PoissonRegressor": PoissonRegressor(),
                "TweedieRegressor": TweedieRegressor(),
                "GammaRegressor": GammaRegressor(),
                
                # Tree-based Models
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "ExtraTreeRegressor": ExtraTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "ExtraTreesRegressor": ExtraTreesRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "HistGradientBoosting": HistGradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "BaggingRegressor": BaggingRegressor(),
                "XGBoost": xgb.XGBRegressor(),
                
                # SVM Models
                "SVR": SVR(),
                "NuSVR": NuSVR(),
                "LinearSVR": LinearSVR(),
                
                # Other Models
                "KNN": KNeighborsRegressor(),
                "KernelRidge": KernelRidge(),
                "GaussianProcess": GaussianProcessRegressor(),
                "MLPRegressor": MLPRegressor(max_iter=1000),
                "DummyRegressor": DummyRegressor(),
                "TransformedTargetRegressor": TransformedTargetRegressor(regressor=LinearRegression())
            }
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = lgb.LGBMRegressor(verbose=-1)
            
            total_models = len(models)
            for idx, (name, model) in enumerate(models.items(), 1):
                print(f"STATUS: Training model {idx}/{total_models}: {name}...")
                try:
                    import time
                    start_time = time.time()
                    
                    base_score = None
                    tuned_score = None
                    best_params = None
                    
                    # Train baseline model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    training_time = time.time() - start_time
                    
                    # Baseline metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    base_score = r2
                    
                    # Hyperparameter tuning if requested
                    if with_tuning:
                        tuning_start = time.time()
                        tuned_model, best_params, tuned_score = self.tuning_service.tune_model(
                            model, name, X_train, y_train, problem_type
                        )
                        if tuned_model:
                            model = tuned_model
                            y_pred = model.predict(X_test)
                            
                            # Recalculate metrics with tuned model
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            
                            training_time += time.time() - tuning_start
                    
                    # GEREKSİNİM 2: Grafikler kaldırıldı, sadece metrikler saklanıyor
                    # Save model
                    model_path = FileManager.save_model(model, f"{name.replace(' ', '_')}.joblib")

                    results.append({
                        "model_name": name,
                        "r2_score": r2,
                        "mse": mse,
                        "mae": mae,
                        "rmse": rmse,
                        "training_time": training_time,
                        "base_score": base_score,
                        "tuned_score": tuned_score,
                        "improvement": (tuned_score - base_score) if (tuned_score and base_score) else None,
                        "best_params": best_params,
                        "model_path": model_path,
                        "status": "success"
                    })
                    print(f"✅ {name} trained successfully (R²: {r2:.3f})")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ FAILED: {name} - {error_msg}")
                    print(f"⏭️  Skipping {name} and continuing with next model...")
                    
                    # Add failed model to results with error status
                    results.append({
                        "model_name": name,
                        "r2_score": 0.0,
                        "mse": 0.0,
                        "mae": 0.0,
                        "rmse": 0.0,
                        "training_time": 0.0,
                        "base_score": None,
                        "tuned_score": None,
                        "improvement": None,
                        "best_params": None,
                        "model_path": None,
                        "status": "failed",
                        "error_message": error_msg
                    })

        # GEREKS İNİM 2A: Generate LazyPredict-style Markdown table
        markdown_table = self.generate_model_results_table(results, problem_type)
        
        # Create performance comparison table
        performance_table = []
        for result in results:
            if problem_type == "classification":
                performance_table.append({
                    "Model": result["model_name"],
                    "Accuracy": round(result["accuracy"], 4),
                    "F1 Score": round(result["f1_score"], 4),
                    "Precision": round(result["precision"], 4),
                    "Recall": round(result["recall"], 4),
                    "ROC AUC": round(result["roc_auc"], 4) if result.get("roc_auc") else "N/A",
                    "Training Time": round(result["training_time"], 2),
                    "Base Score": round(result["base_score"], 4) if result.get("base_score") else "N/A",
                    "Tuned Score": round(result["tuned_score"], 4) if result.get("tuned_score") else "N/A",
                    "Improvement": f"{round(result['improvement'] * 100, 2)}%" if result.get("improvement") else "N/A"
                })
            else:  # regression
                performance_table.append({
                    "Model": result["model_name"],
                    "R² Score": round(result["r2_score"], 4),
                    "MAE": round(result["mae"], 4),
                    "MSE": round(result["mse"], 4),
                    "RMSE": round(result["rmse"], 4),
                    "Training Time": round(result["training_time"], 2),
                    "Base Score": round(result["base_score"], 4) if result.get("base_score") else "N/A",
                    "Tuned Score": round(result["tuned_score"], 4) if result.get("tuned_score") else "N/A",
                    "Improvement": f"{round(result['improvement'] * 100, 2)}%" if result.get("improvement") else "N/A"
                })
        
        # Find best model
        if problem_type == "classification":
            best_model = max(results, key=lambda x: x["accuracy"])
            best_metric = "accuracy"
        else:
            best_model = max(results, key=lambda x: x["r2_score"])
            best_metric = "r2_score"
        
        return {
            "problem_type": problem_type,
            "results": results,
            "performance_table": performance_table,
            "markdown_table": markdown_table,  # GEREKS İNİM 2A: LazyPredict-style table
            "best_model": {
                "name": best_model["model_name"],
                "metric": best_metric,
                "score": round(best_model.get("accuracy" if problem_type == "classification" else "r2_score"), 4)
            },
            "train_test_split": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_percentage": round(len(X_train) / (len(X_train) + len(X_test)) * 100, 1),
                "test_percentage": round(len(X_test) / (len(X_train) + len(X_test)) * 100, 1)
            }
        }
