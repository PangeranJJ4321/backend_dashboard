from typing import Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
from app.models.models import User, Project
from app.schemas.projectSchema import ProjectCreate, ProjectUpdate

class ProjectRepository:
    @staticmethod
    def create_project(db: Session, project: ProjectCreate, user_id: UUID) -> Project:
        db_project = Project(**project.dict(), user_id=user_id)
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        return db_project
    
    @staticmethod
    def get_user_projects(db: Session, user_id: UUID, skip: int = 0, limit: int = 100) -> List[Project]:
        return db.query(Project).filter(Project.user_id == user_id).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_project_by_id(db: Session, project_id: UUID, user_id: UUID) -> Optional[Project]:
        return db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
    
    @staticmethod
    def update_project(db: Session, project_id: UUID, user_id: UUID, project_data: ProjectUpdate) -> Optional[Project]:
        project = db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
        if project:
            update_data = project_data.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(project, key, value)
            db.commit()
            db.refresh(project)
        return project
    
    @staticmethod
    def delete_project(db: Session, project_id: UUID, user_id: UUID) -> bool:
        project = db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
        if project:
            db.delete(project)
            db.commit()
            return True
        return False
    
    @staticmethod
    def count_user_projects(db: Session, user_id: UUID) -> int:
        return db.query(Project).filter(Project.user_id == user_id).count()