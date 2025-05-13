import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw movie data to the format required for model training
    
    Args:
        df: Raw dataframe with movie information
        
    Returns:
        Processed dataframe ready for model training
    """
    # Create a copy to avoid modifying original dataframe
    processed_df = df.copy()
    
    # Extract release year and month from release_date
    processed_df['release_date'] = pd.to_datetime(processed_df['release_date'])
    processed_df['release_year'] = processed_df['release_date'].dt.year
    processed_df['release_month'] = processed_df['release_date'].dt.month
    
    # Calculate ROI (Return on Investment)
    processed_df['ROI'] = ((processed_df['revenue'] - processed_df['budget']) / processed_df['budget']) * 100
    
    # Create ROI category based on ROI value
    conditions = [
        (processed_df['ROI'] < -50),
        (processed_df['ROI'] >= -50) & (processed_df['ROI'] < 0),
        (processed_df['ROI'] >= 0) & (processed_df['ROI'] < 100),
        (processed_df['ROI'] >= 100)
    ]
    choices = ['High Risk', 'Medium Risk', 'Low Risk', 'No Risk']
    processed_df['ROI_category'] = np.select(conditions, choices, default='Unknown')
    
    # Create language binary features
    processed_df['lang_en'] = (processed_df['original_language'] == 'en').astype(int)
    processed_df['lang_others'] = (processed_df['original_language'] != 'en').astype(int)
    
    # One-hot encode genres
    if 'genres' in processed_df.columns:
        # Split the genre string and create binary columns
        genre_df = processed_df['genres'].str.get_dummies(sep=', ')
        
        # Handle all genres from the dataset
        all_genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 
            'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
            'TV Movie', 'Thriller', 'War', 'Western'
        ]
        
        for genre in all_genres:
            # Format genre name for column (lowercase, replace spaces with underscores)
            genre_col_name = f'genre_{genre.lower().replace(" ", "_")}'
            
            # Check if the genre exists in our one-hot encoded columns (case-insensitive)
            genre_cols = [col for col in genre_df.columns if col.lower() == genre.lower()]
            
            if genre_cols:
                # Use the first match if multiple exist
                processed_df[genre_col_name] = genre_df[genre_cols[0]]
            else:
                # If genre doesn't exist in this dataset, add a column of zeros
                processed_df[genre_col_name] = 0
    
    # Select and reorder columns to match the desired output format
    # Base columns (non-genre)
    base_columns = [
        'release_year', 'release_month', 'budget', 'popularity', 'runtime',
        'vote_average', 'vote_count', 'lang_en', 'lang_others'
    ]
    
    # Generate genre column names
    genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in all_genres]
    
    # Output columns (target variables at the end)
    target_columns = ['revenue', 'ROI', 'ROI_category']
    
    final_columns = base_columns + genre_columns + target_columns
    
    # Ensure all required columns exist
    for col in final_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    return processed_df[final_columns]

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into features, revenue target, and ROI category target
    
    Args:
        df: Preprocessed dataframe
        
    Returns:
        Tuple of (features_df, revenue_target, roi_category_target)
    """
    # Features for both models (excluding revenue, ROI, and ROI_category)
    # Base columns (non-genre)
    base_columns = [
        'release_year', 'release_month', 'budget', 'popularity', 'runtime',
        'vote_average', 'vote_count', 'lang_en', 'lang_others'
    ]
    
    # Generate genre column names
    all_genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
        'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 
        'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
        'TV Movie', 'Thriller', 'War', 'Western'
    ]
    genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in all_genres]
    
    feature_cols = base_columns + genre_columns
    
    X = df[feature_cols]
    y_regression = df['revenue']
    y_classification = df['ROI_category']
    
    return X, y_regression, y_classification

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("contoh.csv")
    processed_df = preprocess_data(df)
    processed_df.to_csv("processed_data.csv", index=False)
    print("Preprocessing complete. Output saved to processed_data.csv")