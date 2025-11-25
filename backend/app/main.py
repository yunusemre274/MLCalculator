from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .routers import upload, eda, preprocess, train, cleaning, visualization, preprocessing
import os

app = FastAPI(title="Machine Learning Calculator API")

# CORS Configuration - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Set to False when using wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include Routers
app.include_router(upload.router, tags=["Upload"])
app.include_router(cleaning.router, tags=["Cleaning"])
app.include_router(eda.router, tags=["EDA"])
app.include_router(preprocessing.router, tags=["Preprocessing"])  # FIX 3: New preprocessing router
app.include_router(preprocess.router, tags=["Preprocess"])
app.include_router(train.router, tags=["Training"])
app.include_router(visualization.router, tags=["Visualization"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Machine Learning Calculator API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
