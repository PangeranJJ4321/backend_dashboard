from fastapi import APIRouter, Depends, status, Query
from sqlalchemy.orm import Session
from uuid import UUID

from app.core.session import get_db
from app.middleware.security import get_current_user
from app.models.models import User
from app.controllers.projectController import ProjectController
from app.schemas.projectSchema import (
    ProjectCreate, 
    ProjectUpdate, 
    ProjectResponse, 
    ProjectListResponse
)

router = APIRouter(
    prefix="/projects",
    tags=["Projects"]
)


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new project for the current user"""
    return ProjectController.create_project(
        project=project, 
        db=db, 
        current_user=current_user
    )


@router.get("/", response_model=ProjectListResponse)
def get_user_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all projects for the current user with pagination"""
    return ProjectController.get_user_projects(
        skip=skip,
        limit=limit,
        db=db,
        current_user=current_user
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific project by ID"""
    return ProjectController.get_project(
        project_id=project_id,
        db=db,
        current_user=current_user
    )


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: UUID,
    project_data: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a specific project"""
    return ProjectController.update_project(
        project_id=project_id,
        project_data=project_data,
        db=db,
        current_user=current_user
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a specific project"""
    ProjectController.delete_project(
        project_id=project_id,
        db=db,
        current_user=current_user
    )