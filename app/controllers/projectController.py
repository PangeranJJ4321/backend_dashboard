from typing import List, Optional
from fastapi import Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from uuid import UUID

from app.core.session import get_db
from app.middleware.security import get_current_user
from app.models.models import User
from app.repositories.projectRepositories import ProjectRepository
from app.schemas.projectSchema import (
    ProjectCreate, 
    ProjectUpdate, 
    ProjectResponse, 
    ProjectListResponse
)


class ProjectController:
    @staticmethod
    def create_project(
        project: ProjectCreate,
        db: Session,
        current_user: User
    ) -> ProjectResponse:
        """Create a new project for the current user"""
        return ProjectRepository.create_project(db=db, project=project, user_id=current_user.id)
    
    @staticmethod
    def get_user_projects(
        skip: int,
        limit: int,
        db: Session,
        current_user: User
    ) -> ProjectListResponse:
        """Get all projects for the current user with pagination"""
        projects = ProjectRepository.get_user_projects(
            db=db, user_id=current_user.id, skip=skip, limit=limit
        )
        total = ProjectRepository.count_user_projects(db=db, user_id=current_user.id)
        
        return ProjectListResponse(
            items=[ProjectResponse.model_validate(project) for project in projects],
            total=total,
            skip=skip,
            limit=limit
        )

    
    @staticmethod
    def get_project(
        project_id: UUID,
        db: Session,
        current_user: User
    ) -> ProjectResponse:
        """Get a specific project by ID"""
        project = ProjectRepository.get_project_by_id(
            db=db, project_id=project_id, user_id=current_user.id
        )
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        return project
    
    @staticmethod
    def update_project(
        project_id: UUID,
        project_data: ProjectUpdate,
        db: Session,
        current_user: User
    ) -> ProjectResponse:
        """Update a specific project"""
        project = ProjectRepository.update_project(
            db=db, project_id=project_id, user_id=current_user.id, project_data=project_data
        )
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        return project
    
    @staticmethod
    def delete_project(
        project_id: UUID,
        db: Session,
        current_user: User
    ) -> None:
        """Delete a specific project"""
        deleted = ProjectRepository.delete_project(
            db=db, project_id=project_id, user_id=current_user.id
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        return {
            "message" : "Seccesfully delete project"
        }