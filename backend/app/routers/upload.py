from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd

from ..utils.file_manager import FileManager
from ..services.data_processing_service import DataProcessingService

router = APIRouter()

@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided.")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
        file_path = await FileManager.save_upload_file(file)
        df = pd.read_csv(file_path)

        # Automatically remove ID / index-style columns right after upload
        processor = DataProcessingService()
        cleaned_df, cleanup_summary = processor.automatic_column_cleanup(df)
        FileManager.save_dataset(cleaned_df)

        info = {
            "filename": file.filename,
            "rows": cleaned_df.shape[0],
            "columns": cleaned_df.shape[1],
            "column_names": list(cleaned_df.columns),
            "missing_values": cleaned_df.isnull().sum().to_dict(),
            "data_types": cleaned_df.dtypes.astype(str).to_dict(),
            "cleanup_summary": cleanup_summary
        }
        return {"message": "File uploaded successfully", "info": info}
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Error uploading file: {str(e)}")  # Log error to console
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
