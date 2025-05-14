from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import json
from sqlalchemy.orm import Session

from app.schemas.predictionSchema import PredictionCreate, PredictionResponse, PredictionList
from app.repositories.predictionRepositories import PredictionRepository
from app.utils.prediction import preprocess_input, make_prediction


class PredictionService:
    @staticmethod
    def create_prediction(db: Session, prediction_data: PredictionCreate, user_id: UUID) -> PredictionResponse:
        """
        Process prediction data, run ML models, and save results to database.
        
        Args:
            db: Database session
            prediction_data: Input prediction data
            user_id: ID of the user making the prediction
            
        Returns:
            Response with prediction results
        """
        # Preprocess the input data
        preprocessed_data = preprocess_input(
            film_title=prediction_data.film_title,
            release_date=prediction_data.release_date,
            budget=prediction_data.budget,
            genres=prediction_data.genres,
            popularity=prediction_data.popularity,
            runtime=prediction_data.runtime,
            vote_average=prediction_data.vote_average,
            vote_count=prediction_data.vote_count,
            original_language=prediction_data.original_language
        )
        
        # Make predictions using ML models
        predicted_revenue, predicted_roi, risk_level, feature_importance = make_prediction(preprocessed_data)
        
        # Prepare data for database
        db_prediction_data = {
            "project_id": prediction_data.project_id,
            "film_title": prediction_data.film_title,
            "release_date": prediction_data.release_date,
            "budget": prediction_data.budget,
            "predicted_revenue": predicted_revenue,
            "predicted_roi": predicted_roi,
            "risk_level": risk_level,
            "popularity": prediction_data.popularity,
            "runtime": prediction_data.runtime,
            "vote_average": prediction_data.vote_average,
            "vote_count": prediction_data.vote_count,
            "original_language": prediction_data.original_language,
            "feature_importance": feature_importance,
            "genres": prediction_data.genres
        }
        
        # Save prediction to database
        db_prediction = PredictionRepository.create_prediction(
            db=db, 
            prediction_data=db_prediction_data,
            user_id=user_id
        )
        
        # Parse feature importance back from JSON if needed
        if isinstance(db_prediction.feature_importance, str):
            db_prediction.feature_importance = json.loads(db_prediction.feature_importance)
        
        return db_prediction
    
    @staticmethod
    def get_predictions_by_project(
        db: Session, 
        project_id: UUID, 
        user_id: UUID,
        skip: int = 0, 
        limit: int = 100
    ) -> PredictionList:
        """
        Get paginated list of predictions for a project.
        
        Args:
            db: Database session
            project_id: Project ID to filter predictions
            user_id: User ID for authorization
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            
        Returns:
            List of predictions with pagination metadata
        """
        predictions = PredictionRepository.get_predictions_by_project(
            db=db, 
            project_id=project_id, 
            user_id=user_id,
            skip=skip, 
            limit=limit
        )
        
        total = PredictionRepository.count_predictions_by_project(
            db=db, 
            project_id=project_id, 
            user_id=user_id
        )
        
        # Convert feature importance strings to dictionaries
        for prediction in predictions:
            if prediction.feature_importance and isinstance(prediction.feature_importance, str):
                prediction.feature_importance = json.loads(prediction.feature_importance)
        
        return PredictionList(
            items=predictions,
            total=total
        )
    
    @staticmethod
    def get_prediction_by_id(db: Session, prediction_id: UUID, user_id: UUID) -> Optional[PredictionResponse]:
        """
        Get a specific prediction.
        
        Args:
            db: Database session
            prediction_id: ID of the prediction to retrieve
            user_id: ID of the user for authorization
            
        Returns:
            Prediction if found and authorized
        """
        prediction = PredictionRepository.get_prediction_by_id(
            db=db, 
            prediction_id=prediction_id, 
            user_id=user_id
        )
        
        if not prediction:
            return None
        
        # Convert feature importance string to dictionary
        if prediction.feature_importance and isinstance(prediction.feature_importance, str):
            prediction.feature_importance = json.loads(prediction.feature_importance)
        
        return prediction
    
    @staticmethod
    def delete_prediction(db: Session, prediction_id: UUID, user_id: UUID) -> bool:
        """
        Delete a prediction.
        
        Args:
            db: Database session
            prediction_id: ID of the prediction to delete
            user_id: ID of the user for authorization
            
        Returns:
            True if deleted successfully
        """
        return PredictionRepository.delete_prediction(
            db=db, 
            prediction_id=prediction_id, 
            user_id=user_id
        )