from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from app.core.database import engine, Base
from app.routes import authRouter, predictionRouter, projectRouter, userRouter
from app.seeds.seed_genres import seed_genres

load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# seed data gendre 
seed_genres()

app = FastAPI(
    title="Film Investment Risk Prediction API",
    description="API untuk prediksi risiko investasi film berdasarkan data histories",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(authRouter.router, prefix="/api/v1")
# User Routes
app.include_router(userRouter.router, prefix="/api/v1")
# Project Routes
app.include_router(projectRouter.router, prefix="/api/v1")
# Prediction Routes
app.include_router(predictionRouter.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Selamat datang di Film Investment Risk Predictor API",
        "docs": os.getenv("URL_BACKEND")
    }

# 👇 Betulinnya di sini
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
