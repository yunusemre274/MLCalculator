import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .outlier_service import OutlierService

class PreprocessService:
    def __init__(self):
        self.outlier_service = OutlierService()
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def preprocess_data(self, df: pd.DataFrame, target_column: str = None):
        # 1. Handle Missing Values
        # Numerical: Mean imputation
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 0:
            imputer_num = SimpleImputer(strategy='mean')
            df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

        # Categorical: Mode imputation
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

        # 2. Handle Outliers
        df, outlier_report = self.outlier_service.handle_outliers(df, method="auto")

        # 3. Encoding Categorical Variables
        encoding_report = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            encoding_report[col] = "label_encoded"

        # 4. Scaling (only features, not target if classification)
        # If target_column is provided, exclude it from scaling if it's classification (usually)
        # For simplicity, we scale all numerical features except target
        features_to_scale = [col for col in numerical_cols if col != target_column]
        if features_to_scale:
            df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])

        # Create preprocessing summary table
        preprocessing_table = []
        
        # Add encoding info
        for col, method in encoding_report.items():
            preprocessing_table.append({
                "column": col,
                "original_type": "categorical",
                "action": "encoded",
                "method": method,
                "result_shape": f"({len(df)}, 1)"
            })
        
        # Add scaling info
        for col in features_to_scale:
            preprocessing_table.append({
                "column": col,
                "original_type": "numerical",
                "action": "scaled",
                "method": "StandardScaler",
                "result_shape": f"({len(df)}, 1)"
            })
        
        return df, {
            "outlier_report": outlier_report,
            "encoding_report": encoding_report,
            "missing_values_handled": True,
            "scaling_applied": True,
            "preprocessing_summary_table": preprocessing_table,
            "final_shape": {"rows": df.shape[0], "columns": df.shape[1]}
        }
