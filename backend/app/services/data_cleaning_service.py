import pandas as pd
import numpy as np

class DataCleaningService:
    """Service for automatic column removal and data cleaning"""
    
    @staticmethod
    def detect_useless_columns(df: pd.DataFrame):
        """Automatically detect and remove useless columns"""
        removal_report = []
        cols_to_drop = []
        
        for col in df.columns:
            # 1. Unnamed columns
            if 'unnamed' in col.lower() or col.lower().startswith('index'):
                cols_to_drop.append(col)
                removal_report.append({
                    "column": col,
                    "reason": "Unnamed or index column"
                })
                continue
            
            # 2. Constant columns (all same value)
            if df[col].nunique() == 1:
                cols_to_drop.append(col)
                removal_report.append({
                    "column": col,
                    "reason": f"Constant column (only value: {df[col].iloc[0]})"
                })
                continue
            
            # 3. High cardinality categorical (>95% unique)
            if df[col].dtype == 'object':
                unique_pct = df[col].nunique() / len(df) * 100
                if unique_pct > 95:
                    cols_to_drop.append(col)
                    removal_report.append({
                        "column": col,
                        "reason": f"High cardinality ({unique_pct:.1f}% unique values)"
                    })
                    continue
            
            # 4. ID-like columns (numeric with all unique values)
            if df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() == len(df) and 'id' in col.lower():
                    cols_to_drop.append(col)
                    removal_report.append({
                        "column": col,
                        "reason": "ID column (all unique numeric values)"
                    })
                    continue
        
        # Remove duplicates
        duplicates_removed = df.duplicated().sum()
        df_clean = df.drop(columns=cols_to_drop).drop_duplicates()
        
        # Create clean dataset summary table
        summary_table = {
            "original_shape": {"rows": len(df), "columns": len(df.columns)},
            "cleaned_shape": {"rows": len(df_clean), "columns": len(df_clean.columns)},
            "columns_removed": len(cols_to_drop),
            "duplicates_removed": int(duplicates_removed),
            "remaining_columns": list(df_clean.columns)
        }
        
        return df_clean, removal_report, int(duplicates_removed), summary_table
    
    @staticmethod
    def auto_detect_target(df: pd.DataFrame):
        """Auto-detect the most likely target column"""
        candidates = []
        
        for col in df.columns:
            # Common target names
            if any(term in col.lower() for term in ['target', 'label', 'class', 'outcome', 'price', 'salary', 'revenue']):
                candidates.append((col, 10))  # High priority
            
            # Last column often is target
            elif col == df.columns[-1]:
                candidates.append((col, 5))  # Medium priority
            
            # Categorical with few classes
            elif df[col].dtype == 'object' and df[col].nunique() < 10:
                candidates.append((col, 3))  # Low priority
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
