from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.preprocess_service import PreprocessService
from ..utils.file_manager import FileManager

router = APIRouter()
preprocess_service = PreprocessService()

class PreprocessRequest(BaseModel):
    target_column: str = None

@router.post("/preprocess")
async def preprocess_dataset(request: PreprocessRequest):
    try:
        df = FileManager.load_dataset()
        df_clean, report = preprocess_service.preprocess_data(df, request.target_column)
        
        # Save the preprocessed dataset
        FileManager.save_dataset(df_clean)
        
        return {"message": "Preprocessing completed", "report": report}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
