import pandas as pd
import os
import pickle
import numpy as np

def load_model(model_path):
    """Load a trained model from disk"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(input_data_path, output_file_path=None):
    """
    Generate predictions using trained models
    
    Args:
        input_data_path: Path to input data CSV file
        output_file_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    print(f"Loading data from {input_data_path}...")
    df = pd.read_csv(input_data_path)
    
    print("Preprocessing data...")
    processed_df = preprocess_data(df)
    
    # Extract features
    X, _, _ = split_features_target(processed_df)
    
    # Load best models
    # Normally would have a way to determine best models, but for now assume XGBoost
    regression_model = load_model('models/regresi/xgboost_regressor.pkl')
    classification_model = load_model('models/klasifikasi/xgboost_classifier.pkl')
    
    # Generate predictions
    print("Generating predictions...")
    revenue_predictions = regression_model.predict(X)
    risk_predictions = classification_model.predict(X)
    risk_probabilities = classification_model.predict_proba(X)
    
    # Create result dataframe
    results_df = df.copy()
    results_df['predicted_revenue'] = revenue_predictions
    results_df['predicted_risk'] = risk_predictions
    
    # Calculate predicted ROI
    results_df['predicted_roi'] = ((results_df['predicted_revenue'] - results_df['budget']) / results_df['budget']) * 100
    
    # Create feature importance dictionary
    # For a real implementation, would extract feature importance from models
    feature_importance = {
        'budget': 0.35,
        'popularity': 0.25,
        'runtime': 0.15,
        'vote_average': 0.10,
        'vote_count': 0.05,
        'genre': 0.10
    }
    
    results_df['feature_importance'] = str(feature_importance)
    
    # Extract genres for display
    if 'genres' in df.columns:
        results_df['genres_list'] = df['genres']
    
    # Save predictions if output path is provided
    if output_file_path:
        print(f"Saving predictions to {output_file_path}...")
        results_df.to_csv(output_file_path, index=False)
    
    return results_df

def format_for_api(prediction_df):
    """
    Format prediction results for API response
    
    Args:
        prediction_df: DataFrame with predictions
        
    Returns:
        List of dictionaries suitable for API response
    """
    formatted_results = []
    
    for _, row in prediction_df.iterrows():
        # Parse genres
        genres = []
        if 'genres_list' in row and isinstance(row['genres_list'], str):
            genres = [genre.strip() for genre in row['genres_list'].split(',')]
        
        # Parse feature importance
        feature_importance = {}
        if 'feature_importance' in row and isinstance(row['feature_importance'], str):
            try:
                feature_importance = eval(row['feature_importance'])
            except:
                feature_importance = {}
        
        # Format result
        result = {
            'film_title': row.get('title', ''),
            'release_date': row.get('release_date', ''),
            'budget': row.get('budget', 0),
            'predicted_revenue': row.get('predicted_revenue', 0),
            'predicted_roi': row.get('predicted_roi', 0),
            'risk_level': row.get('predicted_risk', ''),
            'genres': genres,
            'popularity': row.get('popularity', 0),
            'runtime': row.get('runtime', 0),
            'vote_average': row.get('vote_average', 0),
            'vote_count': row.get('vote_count', 0),
            'original_language': row.get('original_language', ''),
            'feature_importance': feature_importance
        }
        
        formatted_results.append(result)
    
    return formatted_results

if __name__ == "__main__":
    # Test prediction with sample data
    input_path = "contoh.csv"
    output_path = "prediction_results.csv"
    
    predictions = predict(input_path, output_path)
    formatted_results = format_for_api(predictions)
    
    print("\nPrediction Results (First record):")
    print(f"Title: {formatted_results[0]['film_title']}")
    print(f"Budget: ${formatted_results[0]['budget']:,}")
    print(f"Predicted Revenue: ${formatted_results[0]['predicted_revenue']:,.2f}")
    print(f"Predicted ROI: {formatted_results[0]['predicted_roi']:.2f}%")
    print(f"Risk Level: {formatted_results[0]['risk_level']}")
    print(f"Genres: {', '.join(formatted_results[0]['genres'])}")