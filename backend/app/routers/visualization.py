from fastapi import APIRouter, HTTPException, Query
from ..services.data_processing_service import DataProcessingService
from ..utils.file_manager import FileManager

router = APIRouter()
processing_service = DataProcessingService()

@router.get("/correlation_analysis")
async def correlation_analysis(threshold: float = Query(default=0.8, ge=0.0, le=1.0)):
    """
    Analyzes correlations and generates interactive plots for highly correlated pairs.
    
    Args:
        threshold (float): Correlation threshold (0.0 to 1.0)
        
    Returns:
        JSON with correlation pairs, plots, and heatmap
    """
    try:
        df = FileManager.load_dataset()
        
        results = processing_service.conditional_correlation_plotting(df, threshold)
        
        return {
            "message": "Correlation analysis completed",
            "results": results
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distribution_plots")
async def get_distribution_plots():
    """
    Generates interactive distribution plots for all numerical columns.
    """
    try:
        df = FileManager.load_dataset()
        
        plots = processing_service.generate_distribution_plots(df)
        
        return {
            "message": "Distribution plots generated",
            "plots": plots,
            "total_plots": len(plots)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categorical_plots")
async def get_categorical_plots(max_categories: int = Query(default=20, ge=5, le=100)):
    """
    Generates interactive bar plots for categorical columns.
    
    Args:
        max_categories (int): Maximum number of categories to display per column
    """
    try:
        df = FileManager.load_dataset()
        
        plots = processing_service.generate_categorical_plots(df, max_categories)
        
        return {
            "message": "Categorical plots generated",
            "plots": plots,
            "total_plots": len(plots)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/missing_values_report")
async def get_missing_values_report():
    """
    Generates comprehensive missing values analysis and visualization.
    """
    try:
        df = FileManager.load_dataset()
        
        report = processing_service.generate_missing_values_report(df)
        
        return {
            "message": "Missing values report generated",
            "report": report
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
