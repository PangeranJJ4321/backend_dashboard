import json
from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.models.models import Prediction, Genre, Project, User
from app.schemas.predictionSchema import PredictionCreate
from sqlalchemy import func


class PredictionRepository:
    @staticmethod
    def create_prediction(db: Session, prediction_data: Dict[str, Any], user_id: UUID) -> Prediction:
        """
        Create a new prediction in the database.
        
        Args:
            db: Database session
            prediction_data: Prediction data including results from model
            user_id: ID of the user making the prediction
            
        Returns:
            Created prediction object
        """
        # First, verify the project belongs to the user
        project = db.query(Project).filter(
            and_(
                Project.id == prediction_data["project_id"],
                Project.user_id == user_id
            )
        ).first()
        
        if not project:
            raise ValueError("Project not found or doesn't belong to the user")
        
        # Extract genres to handle the many-to-many relationship
        genre_names = prediction_data.pop("genres", [])
        
        # Handle feature importance - convert to JSON string
        if "feature_importance" in prediction_data and isinstance(prediction_data["feature_importance"], dict):
            prediction_data["feature_importance"] = json.dumps(prediction_data["feature_importance"])
            
        # Create prediction object
        prediction = Prediction(**prediction_data)
        
        # Add genres (get existing or create new ones)
        for genre_name in genre_names:
            genre = db.query(Genre).filter(Genre.name.ilike(genre_name)).first()
            if not genre:
                genre = Genre(name=genre_name)
                db.add(genre)
                db.flush()  # Flush to get the ID
            
            prediction.genres.append(genre)
        
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction
    
    @staticmethod
    def get_predictions_by_project(
        db: Session, 
        project_id: UUID, 
        user_id: UUID,
        skip: int = 0, 
        limit: int = 100
    ) -> List[Prediction]:
        """
        Get predictions for a specific project that belongs to the user.
        
        Args:
            db: Database session
            project_id: Project ID to filter predictions
            user_id: User ID to verify ownership
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            
        Returns:
            List of prediction objects
        """
        # Verify project belongs to user
        project = db.query(Project).filter(
            and_(
                Project.id == project_id,
                Project.user_id == user_id
            )
        ).first()
        
        if not project:
            raise ValueError("Project not found or doesn't belong to the user")
        
        # Get predictions for the project
        predictions = db.query(Prediction).filter(
            Prediction.project_id == project_id
        ).order_by(
            Prediction.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return predictions
    
    @staticmethod
    def count_predictions_by_project(db: Session, project_id: UUID, user_id: UUID) -> int:
        """Count total predictions for a project"""
        # Verify project belongs to user
        project = db.query(Project).filter(
            and_(
                Project.id == project_id,
                Project.user_id == user_id
            )
        ).first()
        
        if not project:
            raise ValueError("Project not found or doesn't belong to the user")
        
        return db.query(func.count()).select_from(Prediction).filter(Prediction.project_id == project_id).scalar()

    
    @staticmethod
    def get_prediction_by_id(db: Session, prediction_id: UUID, user_id: UUID) -> Optional[Prediction]:
        """
        Get a specific prediction that belongs to one of the user's projects.
        
        Args:
            db: Database session
            prediction_id: ID of the prediction to retrieve
            user_id: ID of the user
            
        Returns:
            Prediction object if found and authorized, None otherwise
        """
        # Join to verify that this prediction belongs to a project owned by this user
        prediction = db.query(Prediction).join(
            Project, Prediction.project_id == Project.id
        ).filter(
            and_(
                Prediction.id == prediction_id,
                Project.user_id == user_id
            )
        ).first()
        
        return prediction
    
    @staticmethod
    def delete_prediction(db: Session, prediction_id: UUID, user_id: UUID) -> bool:
        """
        Delete a prediction if it belongs to the user's project.
        
        Args:
            db: Database session
            prediction_id: ID of the prediction to delete
            user_id: ID of the user
            
        Returns:
            True if deleted, False if not found or unauthorized
        """
        prediction = PredictionRepository.get_prediction_by_id(db, prediction_id, user_id)
        
        if not prediction:
            return False
        
        db.delete(prediction)
        db.commit()
        return True