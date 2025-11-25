from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ..services.model_training_service import ModelTrainingService
from ..utils.file_manager import FileManager
import os

router = APIRouter()
training_service = ModelTrainingService()

class TrainRequest(BaseModel):
    target_column: str
    with_tuning: bool = False  # Enable hyperparameter tuning

@router.post("/train_all_models")
async def train_all_models(request: TrainRequest):
    """
    STRICT RULE: This endpoint returns ONLY tabular results.
    NO visualizations (Confusion Matrix, ROC Curve, etc.) are generated or returned.
    
    Output:
    - Model performance table (Markdown format)
    - Model metrics (Accuracy, F1, R², etc.)
    - Best model information
    - Model file paths for download
    
    ❌ EXCLUDED: All Plotly/Matplotlib visualizations
    """
    try:
        df = FileManager.load_dataset()
        if request.target_column not in df.columns:
             raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found in dataset.")
        
        print("=" * 80)
        print("EXECUTING: train_all_models() - TRAIN MODELS TAB")
        print("=" * 80)
        print("✅ Training models...")
        print("✅ Generating performance table (Markdown)")
        print("❌ NO Confusion Matrix")
        print("❌ NO ROC Curve")
        print("❌ NO Feature Importance plots")
        print("=" * 80)
             
        results = training_service.train_models(df, request.target_column, request.with_tuning)
        
        print("✅ Training Complete! ONLY table returned (NO graphs).")
        print("=" * 80)
        
        return results
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_model/{model_name}")
async def download_model(model_name: str):
    try:
        # Ensure extension is present
        if not model_name.endswith('.joblib'):
            model_name += '.joblib'
            
        model_path = FileManager.get_model_path(model_name)
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
            
        return FileResponse(model_path, filename=model_name, media_type='application/octet-stream')
        
        # Better: I will update FileManager in the next step to expose get_model_path.
        # For now, I will assume I can access it via the module import if I change the import.
        # from ..utils.file_manager import FileManager, MODELS_DIR
        
        pass # Placeholder, I will write the file content below correctly.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Correct implementation below
from ..utils.file_manager import MODELS_DIR

@router.get("/download_model")
async def download_model_endpoint(filename: str):
    file_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename=filename)
    raise HTTPException(status_code=404, detail="Model not found")
