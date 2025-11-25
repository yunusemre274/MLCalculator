import os
import pandas as pd
import joblib
from fastapi import UploadFile

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class FileManager:
    @staticmethod
    async def save_upload_file(file: UploadFile) -> str:
        file_path = os.path.join(DATA_DIR, "dataset.csv")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        return file_path

    @staticmethod
    def get_dataset_path() -> str:
        return os.path.join(DATA_DIR, "dataset.csv")

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        path = FileManager.get_dataset_path()
        if not os.path.exists(path):
            raise FileNotFoundError("Dataset not found. Please upload a dataset first.")
        return pd.read_csv(path)

    @staticmethod
    def save_dataset(df: pd.DataFrame):
        path = FileManager.get_dataset_path()
        df.to_csv(path, index=False)

    @staticmethod
    def get_image_path(filename: str) -> str:
        return os.path.join(IMAGES_DIR, filename)
    
    @staticmethod
    def get_relative_image_path(filename: str) -> str:
        return f"/static/images/{filename}"

    @staticmethod
    def save_model(model, filename: str):
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, path)
        return path

    @staticmethod
    def load_model(filename: str):
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError("Model not found.")
        return joblib.load(path)

    @staticmethod
    def get_model_path(filename: str) -> str:
        return os.path.join(MODELS_DIR, filename)
