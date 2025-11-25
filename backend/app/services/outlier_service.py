import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

class OutlierService:
    def handle_outliers(self, df: pd.DataFrame, method="auto"):
        """
        Handle outliers using Winsorization or IQR trimming.
        method: 'auto', 'winsorize', 'iqr'
        """
        df_clean = df.copy()
        numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        
        report = {}

        for col in numerical_cols:
            if method == "auto":
                # Simple heuristic: if skewness is high, use winsorization, else IQR
                skewness = df_clean[col].skew()
                if abs(skewness) > 1:
                    current_method = "winsorize"
                else:
                    current_method = "iqr"
            else:
                current_method = method

            if current_method == "winsorize":
                # Apply winsorization (capping at 5th and 95th percentiles)
                df_clean[col] = winsorize(df_clean[col], limits=[0.05, 0.05])
                report[col] = "winsorized"
            
            elif current_method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap values instead of removing rows to preserve data size
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
                report[col] = "iqr_capped"

        return df_clean, report
