from fastapi import APIRouter, HTTPException, Query
from ..services.eda_service import EDAService
from ..services.eda_comprehensive_service import EDAComprehensiveService
from ..utils.file_manager import FileManager

router = APIRouter()
eda_service = EDAService()
eda_comprehensive_service = EDAComprehensiveService()

@router.get("/run_eda")
async def run_eda():
    """
    Legacy EDA endpoint - now redirects to comprehensive EDA service.
    Uses robust error handling and data cleaning.
    """
    try:
        df = FileManager.load_dataset()
        
        # Use the new comprehensive service with error handling
        results = eda_comprehensive_service.run_complete_eda(df, correlation_threshold=0.8)
        
        return {
            "message": "EDA completed successfully (using comprehensive service)",
            "results": results
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Log the full error for debugging
        import traceback
        print(f"❌ ERROR in /run_eda:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run_complete_eda")
async def run_complete_eda(correlation_threshold: float = Query(default=0.8, ge=0.0, le=1.0)):
    """
    FIX 1: Complete EDA with integrated correlation analysis.
    
    This endpoint executes the full EDA workflow including:
    1. Dataset information
    2. Missing values analysis (with Markdown table - FIX 2)
    3. Basic statistics
    4. Data types summary
    5. Correlation matrix
    6. High correlation scatter plots (correlation >= threshold)
    7. Distribution plots
    8. Categorical plots
    
    The correlation analysis happens HERE in the EDA stage, NOT during training.
    
    Args:
        correlation_threshold (float): Threshold for identifying high correlations (default: 0.8)
    
    Returns:
        Complete EDA results with all analyses and visualizations
    """
    try:
        df = FileManager.load_dataset()
        
        results = eda_comprehensive_service.run_complete_eda(df, correlation_threshold)
        
        return {
            "message": "Complete EDA finished successfully",
            "results": results,
            "correlation_threshold": correlation_threshold
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
