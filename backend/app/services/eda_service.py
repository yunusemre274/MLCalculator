import pandas as pd
import numpy as np
from ..utils.graphs import GraphGenerator

class EDAService:
    def perform_eda(self, df: pd.DataFrame):
        summary = df.describe().to_dict()
        missing_values = df.isnull().sum().to_dict()
        data_types = df.dtypes.astype(str).to_dict()
        
        # Outlier Detection Summary (IQR based)
        outlier_summary = {}
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_summary[col] = int(outliers_count)

        # Generate plots
        plots = {}
        
        # Correlation matrix
        plots['correlation_matrix'] = GraphGenerator.plot_correlation_matrix(df)
        
        # Histograms for numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        plots['histograms'] = []
        for col in numerical_cols:
            # Limit to first 10 numerical columns to avoid too many plots if dataset is huge
            if len(plots['histograms']) < 10:
                plots['histograms'].append({
                    'column': col,
                    'url': GraphGenerator.plot_histogram(df, col)
                })

        # Countplots for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        plots['countplots'] = []
        for col in categorical_cols:
             if len(plots['countplots']) < 10:
                plots['countplots'].append({
                    'column': col,
                    'url': GraphGenerator.plot_countplot(df, col)
                })
        
        # Pairplot for numerical columns (first 5)
        if len(numerical_cols) > 1:
            pairplot_cols = numerical_cols[:min(5, len(numerical_cols))]
            plots['pairplot'] = GraphGenerator.plot_pairplot(df, pairplot_cols)
        
        # High correlation pairs (>= 0.8) - generate scatter plots
        correlation_plots = []
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            for i in range(len(numerical_cols)):
                for j in range(i+1, len(numerical_cols)):
                    col1 = numerical_cols[i]
                    col2 = numerical_cols[j]
                    corr_value = abs(corr_matrix.iloc[i, j])
                    
                    # Only plot if correlation >= 0.8 and columns are different
                    if corr_value >= 0.8 and col1 != col2:
                        scatter_url = GraphGenerator.plot_scatter(df, col1, col2, corr_value)
                        correlation_plots.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(float(corr_value), 3),
                            "plot_url": scatter_url
                        })
        
        # Correlation matrix as table
        correlation_table = []
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            for col in numerical_cols:
                row_data = {"column": col}
                for other_col in numerical_cols:
                    row_data[other_col] = round(float(corr_matrix.loc[col, other_col]), 3)
                correlation_table.append(row_data)

        return {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "summary": summary,
            "missing_values_dict": missing_values,
            "missing_values_table": missing_table,
            "correlation_table": correlation_table,
            "high_correlation_pairs": correlation_plots,
            "data_types": data_types,
            "duplicates": duplicates,
            "unique_counts": unique_counts,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "skewness": skewness,
            "outlier_summary": outlier_summary,
            "plots": plots
        }
