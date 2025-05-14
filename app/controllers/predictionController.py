from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session
import logging

from app.schemas.predictionSchema import PredictionCreate, PredictionResponse, PredictionList
from app.services.prediction_service import PredictionService

# Setup logger
logger = logging.getLogger(__name__)

class PredictionController:
    """
    Controller class for handling film prediction operations.
    Acts as an intermediary between routes and services.
    """
    
    @staticmethod
    def create_prediction(
        db: Session, 
        prediction_data: PredictionCreate, 
        user_id: UUID
    ) -> PredictionResponse:
        """
        Process a new prediction request.
        
        Args:
            db: Database session
            prediction_data: Input prediction data from request
            user_id: ID of authenticated user
            
        Returns:
            Processed prediction with results
            
        Raises:
            ValueError: If input data is invalid or project doesn't belong to user
            Exception: For other processing errors
        """
        try:
            logger.info(f"Creating prediction for project {prediction_data.project_id}")
            result = PredictionService.create_prediction(
                db=db,
                prediction_data=prediction_data,
                user_id=user_id
            )
            logger.info(f"Successfully created prediction {result.id}")
            return result
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error creating prediction: {str(e)}", exc_info=True)
            raise
    
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
            user_id: ID of authenticated user
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            
        Returns:
            List of predictions with pagination metadata
            
        Raises:
            ValueError: If project doesn't belong to user
        """
        try:
            logger.info(f"Retrieving predictions for project {project_id}")
            predictions = PredictionService.get_predictions_by_project(
                db=db,
                project_id=project_id, 
                user_id=user_id,
                skip=skip, 
                limit=limit
            )
            logger.info(f"Retrieved {len(predictions.items)} predictions")
            return predictions
        except ValueError as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            raise
    
    @staticmethod
    def get_prediction_by_id(
        db: Session, 
        prediction_id: UUID, 
        user_id: UUID
    ) -> Optional[PredictionResponse]:
        """
        Get a specific prediction by ID.
        
        Args:
            db: Database session
            prediction_id: ID of the prediction to retrieve
            user_id: ID of authenticated user
            
        Returns:
            Prediction details if found and authorized
            
        Raises:
            ValueError: If prediction doesn't belong to user's project
        """
        logger.info(f"Retrieving prediction {prediction_id}")
        prediction = PredictionService.get_prediction_by_id(
            db=db,
            prediction_id=prediction_id, 
            user_id=user_id
        )
        
        if prediction:
            logger.info(f"Successfully retrieved prediction {prediction_id}")
        else:
            logger.warning(f"Prediction {prediction_id} not found for user {user_id}")
            
        return prediction
    
    @staticmethod
    def delete_prediction(
        db: Session, 
        prediction_id: UUID, 
        user_id: UUID
    ) -> bool:
        """
        Delete a prediction.
        
        Args:
            db: Database session
            prediction_id: ID of the prediction to delete
            user_id: ID of authenticated user
            
        Returns:
            True if deleted successfully, False if not found
        """
        logger.info(f"Attempting to delete prediction {prediction_id}")
        result = PredictionService.delete_prediction(
            db=db,
            prediction_id=prediction_id, 
            user_id=user_id
        )
        
        if result:
            logger.info(f"Successfully deleted prediction {prediction_id}")
        else:
            logger.warning(f"Prediction {prediction_id} not found for deletion")
            
        return result