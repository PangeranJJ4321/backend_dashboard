from datetime import datetime, timezone
import pandas as pd
import os
import sys
import json

# Add the parent directory to the path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.prediction import preprocess_input, make_prediction
from app.utils.prediction_debug import log_model_paths

def test_prediction_flow():
    """
    Test the entire prediction flow from input preprocessing to final prediction.
    """
    print("Testing prediction flow...")
    print("="*50)
    
    # Log model paths for debugging
    log_model_paths()
    
    # Create sample input data
    sample_input = {
        "film_title": "Test Film",
        "release_date": datetime.now(timezone.utc),
        "budget": 10000000,  # $10M budget
        "genres": ["Action", "Adventure", "Science Fiction"],
        "popularity": 150.0,
        "runtime": 120,
        "vote_average": 7.5,
        "vote_count": 1000,
        "original_language": "en"
    }
    
    print("Sample input data:")
    print(json.dumps(
        {k: str(v) if isinstance(v, datetime) else v for k, v in sample_input.items()}, 
        indent=2
    ))
    print("\n")
    
    # Preprocess input
    print("Preprocessing input...")
    df = preprocess_input(
        film_title=sample_input["film_title"],
        release_date=sample_input["release_date"],
        budget=sample_input["budget"],
        genres=sample_input["genres"],
        popularity=sample_input["popularity"],
        runtime=sample_input["runtime"],
        vote_average=sample_input["vote_average"],
        vote_count=sample_input["vote_count"],
        original_language=sample_input["original_language"]
    )
    
    print("Preprocessed data shape:", df.shape)
    print("Preprocessed data columns:", df.columns.tolist())
    print("\n")
    
    # Make prediction
    print("Making prediction...")
    predicted_revenue, predicted_roi, risk_level, feature_importance = make_prediction(df)
    
    print(f"Predicted Revenue: ${predicted_revenue:,}")
    print(f"Predicted ROI: {predicted_roi:.2f}%")
    print(f"Risk Level: {risk_level}")
    
    print("\nTop 5 Feature Importances:")
    top_features = list(feature_importance.items())[:5]
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    print("\nPrediction test completed successfully!")
    print("="*50)
    
    return {
        "predicted_revenue": predicted_revenue,
        "predicted_roi": predicted_roi,
        "risk_level": risk_level,
        "top_features": top_features
    }

if __name__ == "__main__":
    test_prediction_flow()