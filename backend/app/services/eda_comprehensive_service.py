"""
FINAL CORRECTED IMPLEMENTATION - STRICT OUTPUT PLACEMENT
==========================================================

This document demonstrates the STRICT separation of EDA visualizations and Training results.

## RULE 1: EDA Tab Contains ALL Visualizations
## RULE 2: Train Models Tab Contains ONLY Tables (NO GRAPHS!)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .data_processing_service import DataProcessingService


class EDAComprehensiveService:
    """
    STRICT RULE: This service handles ALL visualizations for the EDA tab.
    NO visualization functions should be called during model training.
    
    EDA Tab Output:
    1. Structural Summary (Markdown table)
    2. Missing Values (Markdown table + plot)
    3. Basic Statistics (Markdown table)
    4. Data Types Summary (Markdown table)
    5. Correlation Matrix Heatmap (Plotly)
    6. High Correlation Scatter Plots (Plotly, r >= 0.8)
    7. Distribution Plots (Plotly histograms for ALL numerical columns)
    8. Categorical Plots (Plotly bar charts for ALL categorical columns)
    """
    
    def __init__(self):
        self.data_processor = DataProcessingService()
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert NumPy/Pandas types to Python native types for JSON serialization.
        
        Fixes: TypeError: 'numpy.int64' object is not iterable
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def clean_and_impute_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
        """
        ROBUST DATA CLEANING AND IMPUTATION STRATEGY
        
        This method handles:
        1. Numeric conversion with error handling (strings → NaN)
        2. Column-wise missing value management
        3. High-density null removal (>50% NaN → drop column)
        4. Low-density null removal (<1% NaN → drop rows)
        5. Median imputation (1-50% NaN → fill with median)
        
        Args:
            df (pd.DataFrame): Raw input DataFrame
            
        Returns:
            tuple: (cleaned_df, cleaning_report)
        """
        df_clean = df.copy()
        dropped_columns = []
        dropped_rows_count = 0
        imputed_columns = []
        
        # Step 1: Robust Numeric Conversion
        print("🔧 Step 1: Converting strings to NaN in numeric columns...")
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                # Coerce non-numeric values to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Step 2: Column-wise Missing Value Management
        print("🔧 Step 2: Managing missing values...")
        total_rows = len(df_clean)
        
        for col in df_clean.columns:
            nan_count = df_clean[col].isnull().sum()
            nan_ratio = nan_count / total_rows
            
            if nan_ratio > 0.50:
                # High-Density: Drop column
                print(f"   ❌ Dropping column '{col}' (NaN ratio: {nan_ratio:.2%})")
                dropped_columns.append({
                    "column": col,
                    "reason": f"High NaN density ({nan_ratio:.2%})",
                    "nan_count": nan_count
                })
                df_clean = df_clean.drop(columns=[col])
                
            elif nan_ratio < 0.01 and nan_count > 0:
                # Low-Density: Drop rows
                print(f"   🗑️ Dropping rows with NaN in '{col}' (NaN ratio: {nan_ratio:.2%})")
                before_rows = len(df_clean)
                df_clean = df_clean.dropna(subset=[col])
                dropped_rows_count += before_rows - len(df_clean)
                
            elif nan_ratio >= 0.01 and nan_ratio <= 0.50 and nan_count > 0:
                # Median Imputation
                if df_clean[col].dtype in ['int64', 'float64']:
                    median_value = df_clean[col].median()
                    df_clean[col].fillna(median_value, inplace=True)
                    print(f"   💉 Imputing '{col}' with median ({median_value:.2f})")
                    imputed_columns.append({
                        "column": col,
                        "strategy": "Median Imputation",
                        "nan_ratio": f"{nan_ratio:.2%}"
                    })
        
        # Generate cleaning report
        report_md = "## 🧹 Data Cleaning Report\n\n"
        
        if dropped_columns:
            report_md += "### Dropped Columns (>50% NaN)\n\n"
            report_md += "| Column | Reason | NaN Count |\n"
            report_md += "|--------|--------|-----------|\n"
            for item in dropped_columns:
                report_md += f"| {item['column']} | {item['reason']} | {item['nan_count']} |\n"
            report_md += "\n"
        
        if dropped_rows_count > 0:
            report_md += f"### Dropped Rows (<1% NaN): {dropped_rows_count} rows\n\n"
        
        if imputed_columns:
            report_md += "### Imputed Columns (1-50% NaN)\n\n"
            report_md += "| Column | Strategy | NaN Ratio |\n"
            report_md += "|--------|----------|-----------|\n"
            for item in imputed_columns:
                report_md += f"| {item['column']} | {item['strategy']} | {item['nan_ratio']} |\n"
        
        cleaning_report = {
            "markdown_table": report_md,
            "dropped_columns": dropped_columns,
            "dropped_rows_count": dropped_rows_count,
            "imputed_columns": imputed_columns,
            "original_shape": df.shape,
            "cleaned_shape": df_clean.shape
        }
        
        print(f"✅ Cleaning complete: {df.shape} → {df_clean.shape}")
        return df_clean, cleaning_report
    
    def run_complete_eda(self, df: pd.DataFrame, correlation_threshold: float = 0.8) -> Dict:
        """
        MAIN EDA METHOD - Called from /run_complete_eda endpoint.
        
        ERROR RESOLUTION & ROBUSTNESS:
        - Robust data type handling and filtering
        - Comprehensive NaN handling before plotting
        - Try-except blocks for error recovery
        - Debugging and logging output
        
        Args:
            df (pd.DataFrame): Input DataFrame
            correlation_threshold (float): Threshold for high correlations
            
        Returns:
            Dict: Complete EDA results with ALL tables and visualizations
        """
        
        print("="*80)
        print("DEBUG: EDA process initiated.")
        print("EXECUTING: run_complete_eda() - EDA TAB")
        print("="*80)
        
        eda_results = {
            "status_messages": []  # Track all status updates
        }
        
        try:
            # Step 0: Clean and impute data
            status_msg = "STATUS: Cleaning data and handling missing values..."
            print(f"🔧 {status_msg}")
            eda_results["status_messages"].append(status_msg)
            
            df_cleaned, cleaning_report = self.clean_and_impute_data(df)
            eda_results["cleaning_report"] = cleaning_report
            
            # Step 1: Structural Summary (original data)
            status_msg = "STATUS: Generating structural summary..."
            print(f"✅ {status_msg}")
            eda_results["status_messages"].append(status_msg)
            eda_results["dataset_info"] = self._get_structural_summary(df)
            
            # Step 3: Missing Values Analysis (on original data)
            print("✅ Generating Missing Values Analysis (Table + Plot)")
            try:
                eda_results["missing_values"] = self.data_processor.generate_missing_values_report(df)
            except Exception as e:
                print(f"⚠️ Warning: Missing values analysis failed: {e}")
                eda_results["missing_values"] = {
                    "markdown_table": f"ERROR: Failed to analyze missing values. Reason: {str(e)}",
                    "has_missing_values": False
                }
            
            # Step 4: Basic Statistics (on cleaned data)
            print("✅ Generating Basic Statistics (Table)")
            try:
                eda_results["basic_statistics"] = self._get_basic_statistics(df_cleaned)
            except Exception as e:
                print(f"⚠️ Warning: Basic statistics failed: {e}")
                eda_results["basic_statistics"] = {
                    "markdown_table": f"ERROR: Failed to calculate statistics. Reason: {str(e)}"
                }
            
            # Step 5: Data Types Summary
            print("✅ Generating Data Types Summary (Table)")
            eda_results["data_types_summary"] = self._get_data_types_summary(df)
            
            # Step 6: Correlation Analysis (on cleaned data)
            # CRITICAL: Use cleaned data to avoid NaN crashes
            print("✅ Generating Correlation Matrix (Plotly Heatmap)")
            print("✅ Generating High Correlation Scatter Plots (Plotly)")
            try:
                eda_results["correlation_analysis"] = self.data_processor.conditional_correlation_plotting(
                    df_cleaned, correlation_threshold
                )
            except Exception as e:
                print(f"⚠️ Warning: Correlation analysis failed: {e}")
                eda_results["correlation_analysis"] = {
                    "message": f"ERROR: Failed to draw correlation plots. Reason: {str(e)}",
                    "high_correlation_pairs": [],
                    "total_pairs": 0
                }
            
            # Step 7: Distribution Plots (on cleaned data)
            status_msg = "STATUS: Generating Distribution Graphs..."
            print(f"✅ {status_msg}")
            eda_results["status_messages"].append(status_msg)
            
            try:
                eda_results["distribution_plots"] = self.data_processor.generate_distribution_plots(df_cleaned)
            except Exception as e:
                print(f"⚠️ Warning: Distribution plots failed: {e}")
                eda_results["distribution_plots"] = {}
            
            # Step 8: Categorical Plots (on original data)
            print("✅ Generating Categorical Plots (Plotly Bar Charts)")
            try:
                eda_results["categorical_plots"] = self.data_processor.generate_categorical_plots(df)
            except Exception as e:
                print(f"⚠️ Warning: Categorical plots failed: {e}")
                eda_results["categorical_plots"] = {}
            
            status_msg = "STATUS: EDA has been successfully completed. Visualizations are ready."
            print(f"✅ {status_msg}")
            eda_results["status_messages"].append(status_msg)
            eda_results["status"] = "complete"
            print("=" * 80)
            
        except Exception as e:
            # Global error handler
            print(f"❌ CRITICAL ERROR in run_complete_eda: {e}")
            import traceback
            traceback.print_exc()
            
            eda_results = {
                "status": "error",
                "error_message": f"EDA Failed: {str(e)}",
                "status_messages": ["ERROR: EDA process failed"],
                "structural_summary": f"## ❌ EDA Error\n\nFailed to complete EDA analysis.\n\n**Error:** {str(e)}",
                "plots": []
            }
        
        # Convert all NumPy/Pandas types to JSON-serializable Python types
        print("🔄 Converting results to JSON-serializable format...")
        eda_results = self._convert_to_json_serializable(eda_results)
        print("✅ Conversion complete!")
        
        # FINAL FIX: Standardize output structure for frontend
        print("📦 Standardizing output structure...")
        standardized_output = self._standardize_eda_output(eda_results)
        print("✅ Output standardization complete!")
        
        return standardized_output
    
    def _get_structural_summary(self, df: pd.DataFrame) -> Dict:
        """
        GEREKSİNİM 1A: Tablosal Veri Özeti (Structural Summary)
        
        Returns a comprehensive structural overview as Markdown table.
        """
        total_rows = len(df)
        total_columns = len(df.columns)
        total_cells = total_rows * total_columns
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        duplicate_rows = df.duplicated().sum()
        
        # Generate Markdown table
        markdown = "## 📊 Veri Seti Yapısal Özeti (Structural Summary)\n\n"
        markdown += "| Metrik | Değer |\n"
        markdown += "|--------|-------|\n"
        markdown += f"| Satır Sayısı (Rows) | {total_rows:,} |\n"
        markdown += f"| Kolon Sayısı (Columns) | {total_columns} |\n"
        markdown += f"| Toplam Hücre Sayısı (Total Cells) | {total_cells:,} |\n"
        markdown += f"| Bellek Kullanımı (Memory Usage) | {memory_usage:.2f} MB |\n"
        markdown += f"| Tekrarlanan Satır Sayısı | {duplicate_rows:,} |\n"
        
        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_cells": total_cells,
            "memory_usage_mb": f"{memory_usage:.2f} MB",
            "duplicate_rows": duplicate_rows,
            "markdown_table": markdown,
            "column_names": list(df.columns)
        }
    
    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate basic statistical summary as Markdown table.
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {
                "message": "No numerical columns found for statistical analysis",
                "markdown_table": None
            }
        
        stats_df = df[numerical_cols].describe().T
        stats_df = stats_df.round(2)
        stats_df.insert(0, 'Column', stats_df.index)
        stats_df.reset_index(drop=True, inplace=True)
        
        # Generate Markdown table
        markdown = "## 📈 Basic Statistics Summary\n\n"
        markdown += stats_df.to_markdown(index=False)
        
        return {
            "statistics": stats_df.to_dict('records'),
            "markdown_table": markdown,
            "numerical_columns_count": len(numerical_cols)
        }
    
    def _get_data_types_summary(self, df: pd.DataFrame) -> Dict:
        """
        Summarize data types distribution as Markdown table.
        """
        dtype_counts = df.dtypes.value_counts()
        
        dtype_df = pd.DataFrame({
            'Data Type': dtype_counts.index.astype(str),
            'Column Count': dtype_counts.values
        })
        
        # Generate Markdown table
        markdown = "## 🔤 Data Types Summary\n\n"
        markdown += "| Data Type | Column Count |\n"
        markdown += "|-----------|-------------|\n"
        for _, row in dtype_df.iterrows():
            markdown += f"| {row['Data Type']} | {row['Column Count']} |\n"
        
        # Add detailed column listing by type
        markdown += "\n### Columns by Data Type\n\n"
        for dtype in df.dtypes.unique():
            cols = df.select_dtypes(include=[dtype]).columns.tolist()
            markdown += f"**{dtype}**: {', '.join(f'`{col}`' for col in cols)}\n\n"
        
        return {
            "data_types": dtype_df.to_dict('records'),
            "markdown_table": markdown,
            "total_types": len(dtype_counts)
        }
    
    def _standardize_eda_output(self, eda_results: Dict) -> Dict:
        """
        FINAL FIX: Standardize EDA output structure for universal frontend compatibility.
        
        Separates tables (Markdown strings) and visualizations (Plotly JSON strings).
        
        Returns:
            Dict: Standardized output structure
        """
        # Extract all plots into a unified array
        plots = []
        
        # Add correlation heatmap
        if "correlation_analysis" in eda_results and not eda_results["correlation_analysis"].get("error"):
            corr_analysis = eda_results["correlation_analysis"]
            
            if corr_analysis.get("correlation_heatmap_json"):
                plots.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix Heatmap",
                    "data": corr_analysis["correlation_heatmap_json"]
                })
            
            # Add high correlation scatter plots
            for pair in corr_analysis.get("high_correlation_pairs", []):
                plots.append({
                    "type": "scatter",
                    "title": f"{pair['column1']} vs {pair['column2']}",
                    "correlation": pair["correlation"],
                    "data": pair["plot_json"]
                })
        
        # Add distribution plots
        if "distribution_plots" in eda_results:
            for col_name, plot_json in eda_results["distribution_plots"].items():
                plots.append({
                    "type": "distribution",
                    "title": f"Distribution of {col_name}",
                    "column": col_name,
                    "data": plot_json
                })
        
        # Add categorical plots
        if "categorical_plots" in eda_results:
            for col_name, plot_json in eda_results["categorical_plots"].items():
                plots.append({
                    "type": "categorical",
                    "title": f"Categories in {col_name}",
                    "column": col_name,
                    "data": plot_json
                })
        
        structural_info = eda_results.get("dataset_info", {}) or {}
        structural_rows = []
        if structural_info:
            structural_rows = [
                {"metric": "Rows", "value": structural_info.get("total_rows", 0)},
                {"metric": "Columns", "value": structural_info.get("total_columns", 0)},
                {"metric": "Total Cells", "value": structural_info.get("total_cells", 0)},
                {"metric": "Memory Usage", "value": structural_info.get("memory_usage_mb", "0 MB")},
                {"metric": "Duplicate Rows", "value": structural_info.get("duplicate_rows", 0)}
            ]

        missing_rows = eda_results.get("missing_values", {}).get("missing_table_data", [])
        basic_stats_rows = eda_results.get("basic_statistics", {}).get("statistics", [])

        cleaning_info = eda_results.get("cleaning_report", {}) or {}
        cleaning_sections = {
            "dropped_columns": cleaning_info.get("dropped_columns", []),
            "dropped_rows_count": cleaning_info.get("dropped_rows_count", 0),
            "imputed_columns": cleaning_info.get("imputed_columns", []),
            "original_shape": cleaning_info.get("original_shape"),
            "cleaned_shape": cleaning_info.get("cleaned_shape")
        }

        # Build standardized structure
        standardized = {
            "status": eda_results.get("status", "complete"),
            "status_messages": eda_results.get("status_messages", []),
            
            # Tables (Markdown strings)
            "structural_summary": eda_results.get("dataset_info", {}).get("markdown_table", ""),
            "missing_values_table": eda_results.get("missing_values", {}).get("markdown_table", ""),
            "basic_statistics_table": eda_results.get("basic_statistics", {}).get("markdown_table", ""),
            "data_types_table": eda_results.get("data_types_summary", {}).get("markdown_table", ""),
            "cleaning_report": eda_results.get("cleaning_report", {}).get("markdown_table", ""),
            "structural_summary_rows": structural_rows,
            "column_names": structural_info.get("column_names", []),
            "missing_values_rows": missing_rows,
            "basic_statistics_rows": basic_stats_rows,
            "cleaning_report_details": cleaning_sections,
            
            # Plots (Plotly JSON strings in standardized array)
            "plots": plots,
            
            # Additional metadata
            "total_plots": len(plots),
            "correlation_threshold": eda_results.get("correlation_analysis", {}).get("threshold_used", 0.8),
            "high_correlation_count": eda_results.get("correlation_analysis", {}).get("total_pairs", 0)
        }
        
        return standardized


# ==================== USAGE EXAMPLE FOR EDA ROUTER ====================

"""
Example usage in EDA router:

from app.services.eda_comprehensive_service import EDAComprehensiveService

@router.post("/run_complete_eda")
async def run_complete_eda(correlation_threshold: float = 0.8):
    '''
    Runs complete EDA including correlation analysis.
    FIX 1: Correlation plots are generated HERE in EDA stage.
    '''
    try:
        df = FileManager.load_dataset()
        
        eda_service = EDAComprehensiveService()
        results = eda_service.run_complete_eda(df, correlation_threshold)
        
        return {
            "message": "Complete EDA finished successfully",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
