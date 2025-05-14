import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Define paths to the model files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_DIR = os.path.join(BASE_DIR, "models", "models")

REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, "regresi", "random_forest_regressor.pkl")
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "klasifikasi", "random_forest_classifier.pkl")

print(MODEL_DIR)

# List of all genres to ensure consistent one-hot encoding
ALL_GENRES = [
    "action", "adventure", "animation", "comedy", "crime",
    "documentary", "drama", "family", "fantasy", "history",
    "horror", "music", "mystery", "romance", "science_fiction",
    "tv_movie", "thriller", "war", "western"
]

# List of all languages (simplified to English and others for example)
ALL_LANGUAGES = ["en", "others"]

# List of all features used during training
FEATURES = [
    "release_year", "release_month", "budget", "popularity", "runtime", 
    "vote_average", "vote_count", "lang_en", "lang_others"
] + [f"genre_{genre}" for genre in ALL_GENRES]


def load_models():
    """Load regression and classification models from disk."""
    with open(REGRESSION_MODEL_PATH, 'rb') as f:
        regression_model = pickle.load(f)
    
    with open(CLASSIFICATION_MODEL_PATH, 'rb') as f:
        classification_model = pickle.load(f)
    
    return regression_model, classification_model


def preprocess_input(
    film_title: str,
    release_date: datetime, 
    budget: int,
    genres: List[str],
    popularity: Optional[float] = None,
    runtime: Optional[int] = None,
    vote_average: Optional[float] = None,
    vote_count: Optional[int] = None,
    original_language: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    
    Args:
        film_title: Title of the film (not used in model but recorded)
        release_date: Planned release date
        budget: Production budget in USD
        genres: List of genre names
        popularity: Movie popularity score
        runtime: Movie duration in minutes
        vote_average: Average vote score (0-10)
        vote_count: Number of votes
        original_language: Original language code
        
    Returns:
        DataFrame ready for model prediction
    """
    # Extract year and month from release date
    release_year = release_date.year
    release_month = release_date.month
    
    # Create base data dictionary with default values
    data = {
        "release_year": release_year,
        "release_month": release_month,
        "budget": budget,
        "popularity": popularity if popularity is not None else 0,
        "runtime": runtime if runtime is not None else 0,
        "vote_average": vote_average if vote_average is not None else 0,
        "vote_count": vote_count if vote_count is not None else 0,
    }
    
    # Handle language encoding (simplified to en/others for this example)
    if original_language is None:
        original_language = "en"  # Default to English if not specified
    
    data["lang_en"] = 1 if original_language.lower() == "en" else 0
    data["lang_others"] = 1 if original_language.lower() != "en" else 0
    
    # Handle genre one-hot encoding
    genres_lower = [g.lower().replace(" ", "_") for g in genres]
    for genre in ALL_GENRES:
        data[f"genre_{genre}"] = 1 if genre in genres_lower else 0
    
    # Create DataFrame with correct feature order
    df = pd.DataFrame([data])
    
    # Ensure all required features are present and in the correct order
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[FEATURES]


def make_prediction(df: pd.DataFrame) -> Tuple[int, float, str, Dict[str, float]]:
    """
    Make revenue and ROI predictions based on preprocessed input data.
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        Tuple of (predicted_revenue, predicted_roi, risk_level, feature_importance)
    """
    regression_model, classification_model = load_models()
    
    # Predict revenue using regression model
    predicted_revenue = int(regression_model.predict(df)[0])
    
    # Calculate ROI (Return on Investment)
    budget = df["budget"].values[0]
    predicted_roi = (predicted_revenue - budget) / budget * 100
    
    # Predict risk category using classification model
    risk_level_idx = classification_model.predict(df)[0]
    risk_levels = ["High Risk", "Medium Risk", "Low Risk", "No Risk"]
    risk_level = risk_levels[risk_level_idx] if isinstance(risk_level_idx, int) else risk_level_idx
    
    # Extract feature importance for explanation
    reg_feature_importance = dict(zip(
        FEATURES, 
        regression_model.feature_importances_
    ))
    
    class_feature_importance = dict(zip(
        FEATURES, 
        classification_model.feature_importances_
    ))
    
    # Combine both importance scores with weights
    feature_importance = {}
    for feature in FEATURES:
        feature_importance[feature] = (
            0.6 * reg_feature_importance.get(feature, 0) + 
            0.4 * class_feature_importance.get(feature, 0)
        )
    
    # Sort by importance value
    feature_importance = {
        k: v for k, v in sorted(
            feature_importance.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
    }
    
    return predicted_revenue, predicted_roi, risk_level, feature_importance