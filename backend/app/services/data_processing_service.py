"""
Robust Data Processing and Visualization Service for ML Web Application
This service provides comprehensive data cleaning, analysis, and visualization capabilities
using Pandas, Plotly, and Matplotlib for interactive ML dashboards.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataProcessingService:
    """
    Main service class for data processing and visualization in ML applications.
    Handles automatic column cleanup, correlation analysis, and interactive visualizations.
    """
    
    def __init__(self):
        """Initialize the service with default configurations."""
        self.df = None
        self.cleaned_df = None
        self.removed_columns = []
        self.correlation_threshold = 0.8
        self.preprocessing_strategies = {}  # FIX 3: Track preprocessing strategies per column
        
    # ==================== REQUIREMENT 1: AUTOMATIC COLUMN CLEANUP ====================
    
    def automatic_column_cleanup(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Automatically cleans the DataFrame by removing ID columns and zero-variance columns.
        This runs immediately after data upload.
        
        Args:
            df (pd.DataFrame): Raw uploaded DataFrame
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and cleanup summary report
            
        Example Output:
            {
                "original_shape": {"rows": 1000, "columns": 15},
                "cleaned_shape": {"rows": 1000, "columns": 12},
                "removed_columns": [
                    {"name": "Unnamed: 0", "reason": "Index column"},
                    {"name": "customer_id", "reason": "Unique identifier"},
                    {"name": "constant_col", "reason": "Zero variance"}
                ],
                "summary_table_markdown": "| Metric | Value |\\n|--------|-------|..."
            }
        """
        self.df = df.copy()
        original_shape = df.shape
        removed_columns_detail = []
        
        # Step 1: Identify and remove ID/Index columns
        id_patterns = ['unnamed', 'index', 'id', 'kimlik', '_id', 'identifier']
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for ID patterns in column name
            if any(pattern in col_lower for pattern in id_patterns):
                removed_columns_detail.append({
                    "name": col,
                    "reason": "Identified as index/ID column"
                })
                df = df.drop(columns=[col])
                continue
            
            # Check for unique identifier columns (all values unique)
            if df[col].dtype in ['int64', 'float64', 'object']:
                if df[col].nunique() == len(df):
                    removed_columns_detail.append({
                        "name": col,
                        "reason": "All values unique (100% unique identifier)"
                    })
                    df = df.drop(columns=[col])
                    continue
        
        # Step 2: Remove zero-variance columns (constant columns)
        for col in df.columns:
            if df[col].nunique() == 1:
                unique_val = df[col].iloc[0]
                removed_columns_detail.append({
                    "name": col,
                    "reason": f"Zero variance (constant value: {unique_val})"
                })
                df = df.drop(columns=[col])
        
        # Generate summary report
        cleaned_shape = df.shape
        
        # Create Markdown table for display
        markdown_table = self._generate_cleanup_markdown_table(
            original_shape, 
            cleaned_shape, 
            removed_columns_detail
        )
        
        summary = {
            "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
            "cleaned_shape": {"rows": cleaned_shape[0], "columns": cleaned_shape[1]},
            "columns_removed": len(removed_columns_detail),
            "removed_columns": removed_columns_detail,
            "remaining_columns": list(df.columns),
            "summary_table_markdown": markdown_table
        }
        
        self.cleaned_df = df
        self.removed_columns = removed_columns_detail
        
        return df, summary
    
    def _generate_cleanup_markdown_table(self, original_shape: Tuple, 
                                        cleaned_shape: Tuple, 
                                        removed_columns: List[Dict]) -> str:
        """
        Generates a Markdown formatted table summarizing the cleanup process.
        
        Returns:
            str: Markdown formatted table string
        """
        markdown = "## 🧹 Dataset Cleanup Summary\n\n"
        markdown += "| Metric | Value |\n"
        markdown += "|--------|-------|\n"
        markdown += f"| **Original Rows** | {original_shape[0]:,} |\n"
        markdown += f"| **Original Columns** | {original_shape[1]} |\n"
        markdown += f"| **Cleaned Rows** | {cleaned_shape[0]:,} |\n"
        markdown += f"| **Cleaned Columns** | {cleaned_shape[1]} |\n"
        markdown += f"| **Columns Removed** | {len(removed_columns)} |\n\n"
        
        if removed_columns:
            markdown += "### 🗑️ Removed Columns\n\n"
            markdown += "| Column Name | Removal Reason |\n"
            markdown += "|-------------|----------------|\n"
            for col_info in removed_columns:
                markdown += f"| `{col_info['name']}` | {col_info['reason']} |\n"
        
        return markdown
    
    # ==================== REQUIREMENT 2: CONDITIONAL CORRELATION PLOTTING ====================
    
    def conditional_correlation_plotting(self, df: pd.DataFrame, 
                                        threshold: float = 0.8) -> Dict:
        """
        Identifies highly correlated column pairs and generates interactive scatter plots.
        This runs after the correlation matrix is displayed in the EDA tab.
        
        ROBUSTNESS ENHANCEMENTS:
        - Strict numerical isolation with numeric_only=True
        - NaN handling before correlation calculation
        - Error handling for edge cases
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame
            threshold (float): Correlation threshold (default: 0.8)
            
        Returns:
            Dict: Contains correlation pairs and their plot URLs/data
        """
        try:
            # REQUIREMENT 1: Strict Numerical Isolation
            numerical_df = df.select_dtypes(include=[np.number]).copy()
            
            if numerical_df.shape[1] < 2:
                return {
                    "message": "Insufficient numerical columns for correlation analysis",
                    "high_correlation_pairs": [],
                    "total_pairs": 0
                }
            
            # REQUIREMENT 2: Comprehensive NaN Handling
            # Temporary median imputation for visualization only (not training data)
            # Rationale: df.corr() crashes on NaN values, causing 500 errors
            for col in numerical_df.columns:
                if numerical_df[col].isnull().any():
                    median_val = numerical_df[col].median()
                    if pd.notna(median_val):
                        numerical_df[col].fillna(median_val, inplace=True)
                    else:
                        # If median is NaN, fill with 0
                        numerical_df[col].fillna(0, inplace=True)
            
            # Calculate correlation matrix with numeric_only=True for safety
            corr_matrix = numerical_df.corr(numeric_only=True)
            
            # Find high correlation pairs
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    # Check if correlation exceeds threshold
                    if abs(corr_value) >= threshold:
                        # Generate interactive scatter plot (JSON format)
                        plot_json = self._create_correlation_scatter_plot(
                            df, col1, col2, corr_value
                        )
                        
                        # Determine relationship type
                        if corr_value > 0:
                            relationship = "Strong Positive" if corr_value > 0.9 else "Positive"
                        else:
                            relationship = "Strong Negative" if corr_value < -0.9 else "Negative"
                        
                        high_corr_pairs.append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(float(corr_value), 3),
                            "plot_json": plot_json,
                            "relationship": relationship
                        })
            
            # Generate correlation matrix heatmap (JSON format)
            heatmap_json = self._create_correlation_heatmap(corr_matrix)
            
            result = {
                "high_correlation_pairs": high_corr_pairs,
                "total_pairs": len(high_corr_pairs),
                "correlation_matrix": corr_matrix.to_dict(),
                "correlation_heatmap_json": heatmap_json,
                "threshold_used": threshold
            }
            
            return result
            
        except Exception as e:
            # REQUIREMENT 3: Debugging and Logging Output
            print(f"❌ Error in conditional_correlation_plotting: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": True,
                "message": f"ERROR: Failed to draw correlation plots. Reason: {str(e)}",
                "high_correlation_pairs": [],
                "total_pairs": 0,
                "correlation_heatmap_json": None
            }
    
    def _create_correlation_scatter_plot(self, df: pd.DataFrame, 
                                        col1: str, col2: str, 
                                        correlation: float) -> str:
        """
        Creates an interactive scatter plot for two correlated columns using Plotly.
        
        Returns:
            str: JSON string of the Plotly figure (for frontend rendering)
        """
        # Create scatter plot with trend line
        fig = px.scatter(
            df, 
            x=col1, 
            y=col2,
            trendline="ols",
            title=f"Correlation: {col1} vs {col2} (r = {correlation:.3f})",
            labels={col1: col1, col2: col2},
            template="plotly_white"
        )
        
        # Customize layout
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        fig.update_layout(
            width=700,
            height=500,
            hovermode='closest',
            font=dict(size=12)
        )
        
        # Add correlation annotation
        fig.add_annotation(
            text=f"Correlation: {correlation:.3f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        # FINAL FIX: Return JSON instead of HTML for universal frontend compatibility
        return fig.to_json()
    
    def _create_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> str:
        """
        Creates an interactive correlation heatmap using Plotly.
        
        Returns:
            str: JSON string of the Plotly heatmap (for frontend rendering)
        """
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix Heatmap",
            width=800,
            height=800,
            xaxis={'side': 'bottom'},
            template="plotly_white"
        )
        
        # FINAL FIX: Return JSON instead of HTML for universal frontend compatibility
        return fig.to_json()
    
    # ==================== ADDITIONAL VISUALIZATION METHODS ====================
    
    def generate_distribution_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generates interactive distribution plots for all numerical columns.
        
        Returns:
            Dict[str, str]: Dictionary mapping column names to Plotly JSON strings
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        plots = {}
        
        for col in numerical_cols:
            fig = px.histogram(
                df, 
                x=col,
                title=f"Distribution of {col}",
                nbins=30,
                marginal="box",
                template="plotly_white"
            )
            
            fig.update_layout(
                width=600,
                height=400,
                showlegend=False
            )
            
            # FINAL FIX: Return JSON instead of HTML for universal frontend compatibility
            plots[col] = fig.to_json()
        
        return plots
    
    def generate_categorical_plots(self, df: pd.DataFrame, max_categories: int = 20) -> Dict[str, str]:
        """
        Generates interactive bar plots for categorical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            max_categories (int): Maximum number of categories to display
            
        Returns:
            Dict[str, str]: Dictionary mapping column names to Plotly JSON strings
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        plots = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(max_categories)
            
            fig = px.bar(
                x=value_counts.index.tolist(),
                y=value_counts.values.tolist(),
                title=f"Distribution of {col} (Top {max_categories})",
                labels={'x': col, 'y': 'Count'},
                template="plotly_white"
            )
            
            fig.update_layout(
                width=700,
                height=400,
                xaxis_tickangle=-45
            )
            
            # FINAL FIX: Return JSON instead of HTML for universal frontend compatibility
            plots[col] = fig.to_json()
        
        return plots
    
    def generate_missing_values_report(self, df: pd.DataFrame) -> Dict:
        """
        Generates comprehensive missing values analysis with clean Markdown table output.
        
        FIX 2: Enhanced Tabular Output for Missing Values
        Returns a three-column Markdown table for ALL columns (even those with zero missing values).
        
        Returns:
            Dict: Missing values report with Markdown table and visualization
            
        Example Output:
            {
                "markdown_table": "| Column Name | Missing Count | Missing Percentage |\\n...",
                "has_missing_values": True,
                "total_missing_columns": 3,
                "plot_html": "<plotly_html>"
            }
        """
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        # Create comprehensive DataFrame for ALL columns
        missing_df = pd.DataFrame({
            'Column Name': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        })
        
        # Format percentage with two decimal places and % symbol
        missing_df['Missing Percentage'] = missing_df['Missing Percentage'].apply(
            lambda x: f"{x:.2f}%"
        )
        
        # Generate clean Markdown table
        markdown_table = self._generate_missing_values_markdown(missing_df)
        
        # Identify columns with missing values for visualization
        missing_cols_df = missing_df[missing_df['Missing Count'] > 0].copy()
        missing_cols_df['Missing Percentage Numeric'] = missing_data[missing_data > 0] / len(df) * 100
        
        # Create visualization only if there are missing values
        if not missing_cols_df.empty:
            fig = px.bar(
                missing_cols_df,
                x='Column Name',
                y='Missing Percentage Numeric',
                title="Missing Values Analysis (Columns with Missing Data Only)",
                labels={'Missing Percentage Numeric': 'Missing %', 'Column Name': 'Column Name'},
                template="plotly_white",
                color='Missing Percentage Numeric',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                width=800,
                height=400,
                xaxis_tickangle=-45
            )
            
            plot_html = fig.to_html(include_plotlyjs='cdn', div_id="missing_values")
        else:
            plot_html = None
        
        return {
            "markdown_table": markdown_table,
            "has_missing_values": len(missing_cols_df) > 0,
            "total_missing_columns": len(missing_cols_df),
            "total_columns": len(df.columns),
            "plot_html": plot_html,
            "missing_table_data": missing_df.to_dict('records')
        }
    
    def _generate_missing_values_markdown(self, missing_df: pd.DataFrame) -> str:
        """
        Generates a clean Markdown table for missing values analysis.
        
        Args:
            missing_df (pd.DataFrame): DataFrame with columns: Column Name, Missing Count, Missing Percentage
            
        Returns:
            str: Markdown formatted table
        """
        markdown = "## 📊 Missing Values Analysis\n\n"
        markdown += "| Column Name | Missing Count | Missing Percentage |\n"
        markdown += "|-------------|---------------|--------------------|\n"
        
        for _, row in missing_df.iterrows():
            markdown += f"| {row['Column Name']} | {row['Missing Count']} | {row['Missing Percentage']} |\n"
        
        return markdown


    # ==================== FIX 3: PREPROCESSING STRATEGY TRACKING ====================
    
    def track_preprocessing_strategy(self, column: str, strategy: str):
        """
        Track the preprocessing strategy applied to a specific column.
        
        FIX 3: This method records which preprocessing technique was used for each column.
        
        Args:
            column (str): Column name
            strategy (str): Strategy applied (e.g., "Winsorization", "IQR Method", "Median Imputation")
        """
        self.preprocessing_strategies[column] = strategy
    
    def get_preprocessing_strategies_table(self) -> Dict:
        """
        Returns all applied preprocessing strategies as a Markdown table.
        
        FIX 3: Provides clean tabular output for preprocessing strategies.
        
        Returns:
            Dict: Contains Markdown table and strategy details
            
        Example Output:
            {
                "markdown_table": "| Column | Applied Strategy |\\n|---|---|...",
                "total_strategies": 5,
                "strategies": [{"column": "age", "strategy": "Median Imputation"}]
            }
        """
        if not self.preprocessing_strategies:
            return {
                "markdown_table": "No preprocessing strategies applied yet.",
                "total_strategies": 0,
                "strategies": []
            }
        
        # Create DataFrame for easy formatting
        strategies_df = pd.DataFrame([
            {"Column": col, "Applied Strategy": strategy}
            for col, strategy in self.preprocessing_strategies.items()
        ])
        
        # Generate Markdown table
        markdown = "## 🔧 Preprocessing Strategies Applied\n\n"
        markdown += "| Column | Applied Strategy |\n"
        markdown += "|--------|------------------|\n"
        
        for _, row in strategies_df.iterrows():
            markdown += f"| {row['Column']} | {row['Applied Strategy']} |\n"
        
        return {
            "markdown_table": markdown,
            "total_strategies": len(self.preprocessing_strategies),
            "strategies": strategies_df.to_dict('records')
        }
    
    def apply_missing_value_imputation(self, df: pd.DataFrame, 
                                      strategy_map: Dict[str, str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply missing value imputation with strategy tracking.
        
        FIX 3: Handles missing values and tracks the strategy used for each column.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            strategy_map (Dict[str, str]): Optional mapping of column -> strategy
                Strategies: "mean", "median", "mode", "ffill", "bfill", "constant"
                If None, auto-selects best strategy per column type
        
        Returns:
            Tuple[pd.DataFrame, Dict]: Imputed DataFrame and strategy report
        """
        df_imputed = df.copy()
        
        if strategy_map is None:
            strategy_map = {}
            # Auto-detect best strategy for each column with missing values
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        strategy_map[col] = "median"  # Robust to outliers
                    else:
                        strategy_map[col] = "mode"  # Most frequent for categorical
        
        # Apply strategies and track them
        for col, strategy in strategy_map.items():
            if col not in df.columns:
                continue
                
            if df[col].isnull().sum() == 0:
                continue
            
            if strategy == "mean":
                df_imputed[col].fillna(df[col].mean(), inplace=True)
                self.track_preprocessing_strategy(col, "Mean Imputation")
                
            elif strategy == "median":
                df_imputed[col].fillna(df[col].median(), inplace=True)
                self.track_preprocessing_strategy(col, "Median Imputation")
                
            elif strategy == "mode":
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].iloc[0]
                df_imputed[col].fillna(mode_val, inplace=True)
                self.track_preprocessing_strategy(col, "Mode Imputation")
                
            elif strategy == "ffill":
                df_imputed[col].fillna(method='ffill', inplace=True)
                self.track_preprocessing_strategy(col, "Forward Fill")
                
            elif strategy == "bfill":
                df_imputed[col].fillna(method='bfill', inplace=True)
                self.track_preprocessing_strategy(col, "Backward Fill")
                
            elif strategy == "constant":
                df_imputed[col].fillna(0, inplace=True)
                self.track_preprocessing_strategy(col, "Constant (0) Imputation")
        
        # Generate report
        report = self.get_preprocessing_strategies_table()
        
        return df_imputed, report
    
    def apply_outlier_handling(self, df: pd.DataFrame, 
                               method_map: Dict[str, str] = None,
                               threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply outlier handling with strategy tracking.
        
        FIX 3: Handles outliers and tracks the method used for each column.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            method_map (Dict[str, str]): Optional mapping of column -> method
                Methods: "iqr", "zscore", "winsorization", "clip"
                If None, auto-applies IQR method to numerical columns
            threshold (float): Threshold for IQR method (default: 1.5)
        
        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and strategy report
        """
        df_cleaned = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if method_map is None:
            method_map = {col: "iqr" for col in numerical_cols}
        
        for col, method in method_map.items():
            if col not in df.columns or col not in numerical_cols:
                continue
            
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                self.track_preprocessing_strategy(col, "IQR Method")
                
            elif method == "zscore":
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df_cleaned = df_cleaned[(z_scores < 3)]
                self.track_preprocessing_strategy(col, "Z-Score Method")
                
            elif method == "winsorization":
                from scipy.stats.mstats import winsorize
                df_cleaned[col] = winsorize(df[col], limits=[0.05, 0.05])
                self.track_preprocessing_strategy(col, "Winsorization")
                
            elif method == "clip":
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df_cleaned[col] = df_cleaned[col].clip(lower=lower, upper=upper)
                self.track_preprocessing_strategy(col, "Percentile Clipping (1-99%)")
        
        # Generate report
        report = self.get_preprocessing_strategies_table()
        
        return df_cleaned, report


# ==================== USAGE EXAMPLE ====================

"""
Example usage in FastAPI endpoint:

from app.services.data_processing_service import DataProcessingService

@router.post("/upload_and_clean")
async def upload_and_clean(file: UploadFile):
    # Load data
    df = pd.read_csv(file.file)
    
    # Initialize service
    processor = DataProcessingService()
    
    # Run automatic cleanup
    cleaned_df, cleanup_summary = processor.automatic_column_cleanup(df)
    
    # Save cleaned data
    FileManager.save_dataset(cleaned_df)
    
    return {
        "message": "Dataset cleaned successfully",
        "summary": cleanup_summary,
        "markdown_report": cleanup_summary['summary_table_markdown']
    }

@router.get("/correlation_analysis")
async def correlation_analysis(threshold: float = 0.8):
    df = FileManager.load_dataset()
    
    processor = DataProcessingService()
    correlation_results = processor.conditional_correlation_plotting(df, threshold)
    
    return correlation_results
"""
