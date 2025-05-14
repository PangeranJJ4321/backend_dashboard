from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum


class GenreCreate(BaseModel):
    name: str


class GenreResponse(GenreCreate):
    id: UUID

    class Config:
        from_attributes = True


class RiskLevel(str, Enum):
    HIGH_RISK = "High Risk"
    MEDIUM_RISK = "Medium Risk"
    LOW_RISK = "Low Risk"
    NO_RISK = "No Risk"


class PredictionCreate(BaseModel):
    project_id: UUID
    film_title: str
    release_date: datetime
    budget: int = Field(..., gt=0)
    genres: List[str]
    runtime: Optional[int] = Field(None, gt=0)
    popularity: Optional[float] = Field(None, ge=0)
    vote_average: Optional[float] = Field(None, ge=0, le=10)
    vote_count: Optional[int] = Field(None, ge=0)
    original_language: Optional[str] = None

    @validator('budget')
    def budget_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Budget must be positive')
        return v
    
    @validator('release_date')
    def release_date_not_past(cls, v):
        if v < datetime.now(timezone.utc):
            raise ValueError("Tanggal rilis tidak boleh di masa lalu.")
        return v


class PredictionResponse(BaseModel):
    id: UUID
    project_id: UUID
    film_title: str
    release_date: datetime
    budget: int
    predicted_revenue: int
    predicted_roi: float
    risk_level: str
    popularity: Optional[float] = None
    runtime: Optional[int] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    original_language: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None
    genres: List[GenreResponse]
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionList(BaseModel):
    items: List[PredictionResponse]
    total: int


class FeatureImportance(BaseModel):
    feature: str
    importance: float