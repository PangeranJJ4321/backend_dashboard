from fastapi import Depends
from sqlalchemy.orm import Session  
from app.core.session import get_db
from app.schemas.authSchema import (
    UserCreate, 
    UserResponse, 
    UserLogin, 
    Token,
    PasswordResetRequest,
    PasswordReset
)
from app.services.auth_service import (
    forgot_password_user, 
    login_user, 
    register_user, 
    reset_password_user, 
    verify_email_user
)


def register(user_data: UserCreate, db: Session = Depends(get_db)):
    new_user = register_user(user_data, db)
    # Convert SQLAlchemy model to Pydantic model
    user_response = UserResponse.model_validate(new_user)

    return {
        "message": "Registration successful. Please check your email to verify your account.",
        "user": user_response
    }

def verify_email(token: str, db: Session = Depends(get_db)):

    verified_user, access_token = verify_email_user(token, db)
    user_response = UserResponse.model_validate(verified_user)
    
    return {
        "message": "Email successfully verified",
        "access_token": access_token,
        "token_type": "JWT",
        "user": user_response
    }

def login(user_data: UserLogin, db: Session = Depends(get_db)):
    
    user, access_token = login_user(user_data, db)
    user_response = UserResponse.model_validate(user)
    
    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }

def forgot_password(email_data: PasswordResetRequest, db: Session = Depends(get_db)):
    
    forgot_password_user(email_data, db)
    return {
        "message": "If your email is registered, you will receive a password reset link shortly."
    }

def reset_password(token: str, password_data: PasswordReset, db: Session = Depends(get_db)):
    reset_password_user(token, password_data, db)
    return {
        "message": "Password has been successfully reset"
    }