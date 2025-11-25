from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.data_cleaning_service import DataCleaningService
from ..services.data_processing_service import DataProcessingService
from ..utils.file_manager import FileManager

router = APIRouter()
cleaning_service = DataCleaningService()
processing_service = DataProcessingService()

@router.post("/clean_columns")
async def clean_columns():
    """Automatically detect and remove useless columns"""
    try:
        df = FileManager.load_dataset()
        df_clean, removal_report, duplicates_removed, summary_table = cleaning_service.detect_useless_columns(df)
        
        # Save cleaned dataset
        FileManager.save_dataset(df_clean)
        
        return {
            "message": "Columns cleaned successfully",
            "summary_table": summary_table,
            "removal_report": removal_report
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/advanced_cleanup")
async def advanced_cleanup():
    """
    Advanced automatic cleanup using DataProcessingService.
    Removes ID columns, zero-variance columns, and returns Markdown summary.
    """
    try:
        df = FileManager.load_dataset()
        
        # Run advanced cleanup
        cleaned_df, cleanup_summary = processing_service.automatic_column_cleanup(df)
        
        # Save cleaned dataset
        FileManager.save_dataset(cleaned_df)
        
        return {
            "message": "Advanced cleanup completed successfully",
            "summary": cleanup_summary,
            "markdown_report": cleanup_summary['summary_table_markdown']
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/auto_detect_target")
async def auto_detect_target():
    """Auto-detect the most likely target column"""
    try:
        df = FileManager.load_dataset()
        target = cleaning_service.auto_detect_target(df)
        
        return {
            "suggested_target": target,
            "all_columns": list(df.columns)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
