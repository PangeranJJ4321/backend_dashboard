from typing import Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
from app.models.models import User
from app.schemas.userSchema import UserUpdate

class UserRepository:
    @staticmethod
    def get_user_by_id(db: Session, user_id: UUID) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def update_user(db: Session, user_id: UUID, user_data: UserUpdate) -> User:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            update_data = user_data.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(user, key, value)
            db.commit()
            db.refresh(user)
        return user
    
    @staticmethod
    def update_user_password(db: Session, user_id: UUID, hashed_password: str) -> User:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.password = hashed_password
            user.reset_token = None
            user.token_expires = None
            db.commit()
            db.refresh(user)
        return user

