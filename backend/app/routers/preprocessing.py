"""
FIX 3: Preprocessing Router with Strategy Tracking
This router handles missing value imputation and outlier handling with Markdown table outputs.
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Optional
from pydantic import BaseModel
from ..services.data_processing_service import DataProcessingService
from ..utils.file_manager import FileManager

router = APIRouter()
processing_service = DataProcessingService()


class ImputationRequest(BaseModel):
    """Request model for missing value imputation"""
    strategy_map: Optional[Dict[str, str]] = None
    # strategy_map example: {"age": "median", "name": "mode", "salary": "mean"}


class OutlierRequest(BaseModel):
    """Request model for outlier handling"""
    method_map: Optional[Dict[str, str]] = None
    threshold: float = 1.5
    # method_map example: {"age": "iqr", "salary": "winsorization"}


@router.post("/impute_missing_values")
async def impute_missing_values(request: ImputationRequest = Body(default=None)):
    """
    FIX 3: Apply missing value imputation with strategy tracking.
    
    Returns a Markdown table showing which strategy was applied to each column.
    
    Available strategies:
    - "mean": Mean imputation for numerical columns
    - "median": Median imputation (robust to outliers)
    - "mode": Mode imputation for categorical columns
    - "ffill": Forward fill (use previous value)
    - "bfill": Backward fill (use next value)
    - "constant": Fill with 0
    
    If strategy_map is not provided, auto-selects best strategy:
    - Numerical columns: median
    - Categorical columns: mode
    
    Returns:
        JSON with imputed dataset info and Markdown strategy table
    """
    try:
        df = FileManager.load_dataset()
        
        strategy_map = request.strategy_map if request else None
        
        # Apply imputation with tracking
        df_imputed, report = processing_service.apply_missing_value_imputation(
            df, strategy_map
        )
        
        # Save imputed dataset
        FileManager.save_dataset(df_imputed)
        
        return {
            "message": "Missing values imputed successfully",
            "strategies_applied": report["total_strategies"],
            "markdown_table": report["markdown_table"],
            "strategy_details": report["strategies"],
            "new_shape": {"rows": df_imputed.shape[0], "columns": df_imputed.shape[1]}
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/handle_outliers")
async def handle_outliers(request: OutlierRequest = Body(default=None)):
    """
    FIX 3: Apply outlier handling with strategy tracking.
    
    Returns a Markdown table showing which method was applied to each column.
    
    Available methods:
    - "iqr": Interquartile Range method (clips values beyond Q1-1.5*IQR and Q3+1.5*IQR)
    - "zscore": Z-Score method (removes values with |z-score| > 3)
    - "winsorization": Winsorization (caps extreme values at 5th and 95th percentiles)
    - "clip": Percentile clipping (clips at 1st and 99th percentiles)
    
    If method_map is not provided, applies IQR method to all numerical columns.
    
    Args:
        method_map: Dictionary mapping column names to methods
        threshold: IQR threshold multiplier (default: 1.5)
    
    Returns:
        JSON with cleaned dataset info and Markdown strategy table
    """
    try:
        df = FileManager.load_dataset()
        
        method_map = request.method_map if request and request.method_map else None
        threshold = request.threshold if request else 1.5
        
        # Apply outlier handling with tracking
        df_cleaned, report = processing_service.apply_outlier_handling(
            df, method_map, threshold
        )
        
        # Save cleaned dataset
        FileManager.save_dataset(df_cleaned)
        
        return {
            "message": "Outliers handled successfully",
            "strategies_applied": report["total_strategies"],
            "markdown_table": report["markdown_table"],
            "strategy_details": report["strategies"],
            "new_shape": {"rows": df_cleaned.shape[0], "columns": df_cleaned.shape[1]}
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preprocessing_strategies")
async def get_preprocessing_strategies():
    """
    FIX 3: Get all preprocessing strategies applied so far.
    
    Returns a Markdown table showing all preprocessing steps taken.
    Useful for reviewing what has been done before model training.
    
    Returns:
        JSON with complete preprocessing history as Markdown table
    """
    try:
        report = processing_service.get_preprocessing_strategies_table()
        
        return {
            "message": "Preprocessing strategies retrieved",
            "markdown_table": report["markdown_table"],
            "total_strategies": report["total_strategies"],
            "strategy_details": report["strategies"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset_preprocessing_tracking")
async def reset_preprocessing_tracking():
    """
    Reset preprocessing strategy tracking.
    Useful when starting a new analysis workflow.
    """
    try:
        processing_service.preprocessing_strategies = {}
        
        return {
            "message": "Preprocessing tracking reset successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
