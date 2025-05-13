from datetime import datetime
from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.session import get_db
from app.repositories.models import User
from app.schemas.schemaUser import UserCreate


# get user by id
def get_user_by_id(id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == id).first()
    return user

def find_user_by_email(email: str, db: Session = Depends(get_db)):
    return db.query(User).filter(User.email == email).first()

def create_new_user(user_data: UserCreate, password_hash: str, verification_token: str, 
                   token_expires: datetime, db: Session = Depends(get_db)):
    # Set current time for created_at
    current_time = datetime.now()
    
    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password=password_hash,
        photo=user_data.photo,
        created_at=current_time,
        updated_at=current_time,  # Set initial updated_at same as created_at
        verification_token=verification_token,
        token_expires=token_expires
    )

    # Save to database
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user

def update_user_verification(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.is_verified = True
        user.verification_token = None
        user.token_expires = None
        user.updated_at = datetime.now()
        db.commit()
        db.refresh(user)
    return user

def find_user_by_token(token: str, token_type: str, db: Session = Depends(get_db)):

    if token_type == "verification":
        return db.query(User).filter(User.verification_token == token).first()
    elif token_type == "reset":
        return db.query(User).filter(User.reset_token == token).first()
    return None

def update_user_password(user_id: str, password_hash: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.password = password_hash
        user.reset_token = None
        user.token_expires = None
        user.updated_at = datetime.now()
        db.commit()
        db.refresh(user)
    return user