import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback
from typing import List, Dict, Any, Tuple, Optional

from app.utils.prediction_debug import debug_prediction_process

# Setup logger
logger = logging.getLogger(__name__)

# Configure logger if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Add file handler to log to file
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "prediction.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Define paths to the model files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "models")
REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, "regresi", "random_forest_regressor.pkl")
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "klasifikasi", "random_forest_classifier.pkl")

# Create mock models for fallback if the actual models aren't found
class MockModel:
    """A fallback mock model if real models can't be loaded."""
    
    def __init__(self, is_regression=True):
        self.is_regression = is_regression
        self.feature_importances_ = [0.1] * 28  # Mock feature importances
        
    def predict(self, X):
        """Return mock predictions."""
        if self.is_regression:
            # For regression, predict revenue as 2.5x budget
            return np.array([X["budget"].values[0] * 2.5])
        else:
            # For classification, return "Medium Risk" (idx 1)
            return np.array([1])

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


@debug_prediction_process
def load_models():
    """
    Load regression and classification models from disk.
    
    Returns:
        Tuple of (regression_model, classification_model)
    """
    try:
        logger.info("Loading prediction models")
        
        # Check if model directories exist
        if not os.path.exists(MODEL_DIR):
            logger.warning(f"Models directory not found at {MODEL_DIR}. Creating directory.")
            os.makedirs(MODEL_DIR, exist_ok=True)
            
        regression_dir = os.path.join(MODEL_DIR, "regresi")
        if not os.path.exists(regression_dir):
            logger.warning(f"Regression model directory not found at {regression_dir}. Creating directory.")
            os.makedirs(regression_dir, exist_ok=True)
            
        classification_dir = os.path.join(MODEL_DIR, "klasifikasi")
        if not os.path.exists(classification_dir):
            logger.warning(f"Classification model directory not found at {classification_dir}. Creating directory.")
            os.makedirs(classification_dir, exist_ok=True)
        
        # Check if model files exist
        if not os.path.exists(REGRESSION_MODEL_PATH):
            logger.warning(f"Regression model not found at {REGRESSION_MODEL_PATH}. Using mock model.")
            regression_model = MockModel(is_regression=True)
        else:
            logger.info(f"Loading regression model from {REGRESSION_MODEL_PATH}")
            with open(REGRESSION_MODEL_PATH, 'rb') as f:
                regression_model = pickle.load(f)
            logger.info("Regression model loaded successfully")
        
        if not os.path.exists(CLASSIFICATION_MODEL_PATH):
            logger.warning(f"Classification model not found at {CLASSIFICATION_MODEL_PATH}. Using mock model.")
            classification_model = MockModel(is_regression=False)
        else:
            logger.info(f"Loading classification model from {CLASSIFICATION_MODEL_PATH}")
            with open(CLASSIFICATION_MODEL_PATH, 'rb') as f:
                classification_model = pickle.load(f)
            logger.info("Classification model loaded successfully")
        
        return regression_model, classification_model
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        # Return mock models as fallback
        logger.warning("Using mock models as fallback due to loading error")
        return MockModel(is_regression=True), MockModel(is_regression=False)


@debug_prediction_process
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
    try:
        logger.info(f"Preprocessing input for film: {film_title}")
        
        # Extract year and month from release date
        release_year = release_date.year
        release_month = release_date.month
        
        logger.debug(f"Release date: {release_date}, year: {release_year}, month: {release_month}")
        
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
        
        logger.debug(f"Language: {original_language}, lang_en: {data['lang_en']}, lang_others: {data['lang_others']}")
        
        # Handle genre one-hot encoding
        genres_lower = [g.lower().replace(" ", "_") for g in genres]
        logger.debug(f"Processed genres: {genres_lower}")
        
        for genre in ALL_GENRES:
            data[f"genre_{genre}"] = 1 if genre in genres_lower else 0
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([data])
        
        # Ensure all required features are present and in the correct order
        for feature in FEATURES:
            if feature not in df.columns:
                logger.warning(f"Missing feature {feature}, adding with default value 0")
                df[feature] = 0
        
        logger.info("Input preprocessing completed successfully")
        return df[FEATURES]
    
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Error preprocessing input data: {str(e)}")


@debug_prediction_process
def make_prediction(df: pd.DataFrame) -> Tuple[int, float, str, Dict[str, float]]:
    """
    Make revenue and ROI predictions based on preprocessed input data.
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        Tuple of (predicted_revenue, predicted_roi, risk_level, feature_importance)
    """
    try:
        logger.info("Making prediction with preprocessed data")
        
        regression_model, classification_model = load_models()
        
        # Predict revenue using regression model
        logger.debug("Predicting revenue with regression model")
        predicted_revenue = int(regression_model.predict(df)[0])
        logger.info(f"Predicted revenue: ${predicted_revenue:,}")
        
        # Calculate ROI (Return on Investment)
        budget = df["budget"].values[0]
        predicted_roi = (predicted_revenue - budget) / budget * 100
        logger.info(f"Calculated ROI: {predicted_roi:.2f}%")
        
        # Predict risk category using classification model
        logger.debug("Predicting risk with classification model")
        risk_level_idx = classification_model.predict(df)[0]
        risk_levels = ["High Risk", "Medium Risk", "Low Risk", "No Risk"]
        
        # Handle different types of risk_level_idx
        if isinstance(risk_level_idx, (int, np.integer)) and 0 <= risk_level_idx < len(risk_levels):
            risk_level = risk_levels[risk_level_idx]
        elif isinstance(risk_level_idx, str) and risk_level_idx in risk_levels:
            risk_level = risk_level_idx
        else:
            # Default to Medium Risk if classification fails
            logger.warning(f"Unexpected risk level index: {risk_level_idx}, defaulting to Medium Risk")
            risk_level = "Medium Risk"
        
        logger.info(f"Predicted risk level: {risk_level}")
        
        # Extract feature importance for explanation
        logger.debug("Extracting feature importance")
        if hasattr(regression_model, 'feature_importances_'):
            reg_feature_importance = dict(zip(
                FEATURES, 
                regression_model.feature_importances_
            ))
        else:
            # Create default values if feature_importances_ not available
            logger.warning("Regression model has no feature_importances_, using default values")
            reg_feature_importance = {feature: 0.1 for feature in FEATURES}
        
        if hasattr(classification_model, 'feature_importances_'):
            class_feature_importance = dict(zip(
                FEATURES, 
                classification_model.feature_importances_
            ))
        else:
            # Create default values if feature_importances_ not available
            logger.warning("Classification model has no feature_importances_, using default values")
            class_feature_importance = {feature: 0.1 for feature in FEATURES}
        
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
        
        logger.info("Prediction completed successfully")
        return predicted_revenue, predicted_roi, risk_level, feature_importance
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Provide fallback predictions
        logger.warning("Using fallback predictions due to error")
        budget = df["budget"].values[0] if not df.empty else 1000000
        fallback_revenue = int(budget * 2.5)  # Assume 2.5x return
        fallback_roi = 150.0  # 150% ROI
        fallback_risk = "Medium Risk"  # Medium risk 
        fallback_importance = {feature: 0.1 for feature in FEATURES}
        
        # Sort feature importance
        fallback_importance = {
            k: v for k, v in sorted(
                fallback_importance.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        }
        
        logger.info(f"Fallback prediction: Revenue: ${fallback_revenue:,}, ROI: {fallback_roi:.2f}%, Risk: {fallback_risk}")
        return fallback_revenue, fallback_roi, fallback_risk, fallback_importance