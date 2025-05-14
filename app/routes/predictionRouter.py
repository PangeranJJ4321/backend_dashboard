from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
import logging

from app.core.session import get_db
from app.middleware.security import get_current_user
from app.models.models import User
from app.schemas.predictionSchema import PredictionCreate, PredictionResponse, PredictionList
from app.controllers.predictionController import PredictionController

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
    responses={404: {"description": "Not found"}}
)


@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def create_prediction(
    prediction_data: PredictionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new prediction for a film project.
    
    Takes film details and returns revenue prediction and risk assessment.
    """
    try:
        result = PredictionController.create_prediction(
            db=db,
            prediction_data=prediction_data,
            user_id=current_user.id
        )
        return result
    except ValueError as e:
        logger.warning(f"Bad request in create_prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in create_prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction processing"
        )


@router.get("/project/{project_id}", response_model=PredictionList)
def get_predictions_by_project(
    project_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all predictions for a specific project.
    
    Returns paginated list of predictions with their results.
    """
    try:
        predictions = PredictionController.get_predictions_by_project(
            db=db,
            project_id=project_id,
            user_id=current_user.id,
            skip=skip,
            limit=limit
        )
        return predictions
    except ValueError as e:
        logger.warning(f"Not found in get_predictions_by_project: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_predictions_by_project: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving predictions"
        )


@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(
    prediction_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific prediction by ID.
    
    Returns detailed information about a single prediction.
    """
    try:
        prediction = PredictionController.get_prediction_by_id(
            db=db,
            prediction_id=prediction_id,
            user_id=current_user.id
        )
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found"
            )
        
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the prediction"
        )


@router.delete("/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prediction(
    prediction_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a prediction.
    
    Removes a prediction from the database.
    """
    try:
        deleted = PredictionController.delete_prediction(
            db=db,
            prediction_id=prediction_id,
            user_id=current_user.id
        )
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found"
            )
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the prediction"
        )