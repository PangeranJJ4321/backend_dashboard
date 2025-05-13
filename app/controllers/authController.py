from fastapi import Depends, HTTPException, status
from datetime import timedelta, datetime
import uuid
import os
from dotenv import load_dotenv

from sqlalchemy.orm import Session  

from app.core.session import get_db
from app.middleware.security import (
    create_access_token,
    verify_password,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.repositories.userRepositorie import (
    create_new_user, 
    find_user_by_email,
    get_user_by_id,
    update_user_verification,
    find_user_by_token,
    update_user_password
)
from app.schemas.schemaUser import (
    UserCreate, 
    UserResponse, 
    UserLogin, 
    Token,
    PasswordResetRequest,
    PasswordReset
)
from app.services.email_service import EmailService

# Load environment variables
load_dotenv()
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    existing_user = find_user_by_email(user_data.email, db)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create verification token
    verification_token = str(uuid.uuid4())
    token_expires = datetime.now() + timedelta(hours=24)

    # Create user with hashed password (is_verified=False by default)
    password_hash = get_password_hash(user_data.password)
    new_user = create_new_user(
        user_data, 
        password_hash, 
        verification_token, 
        token_expires, 
        db
    )

    # Send verification email
    verification_link = f"{FRONTEND_URL}/verify-email?token={verification_token}"
    EmailService.send_verification_email(
        to_email=new_user.email,
        user_name=new_user.name,
        verification_link=verification_link
    )

    # Convert SQLAlchemy model to Pydantic model
    user_response = UserResponse.model_validate(new_user)

    return {
        "message": "Registration successful. Please check your email to verify your account.",
        "user": user_response
    }

def verify_email(token: str, db: Session = Depends(get_db)):
    # Find user by verification token
    user = find_user_by_token(token, "verification", db)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    # Check if token is expired
    if user.token_expires < datetime.now():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired"
        )
    
    # Update user verification status
    verified_user = update_user_verification(user.id, db)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(verified_user.id)},
        expires_delta=access_token_expires
    )
    
    user_response = UserResponse.model_validate(verified_user)
    
    # Send welcome email
    EmailService.send_welcome_email(
        to_email=verified_user.email,
        user_name=verified_user.name
    )
    
    return {
        "message": "Email successfully verified",
        "access_token": access_token,
        "token_type": "JWT",
        "user": user_response
    }

def login(user_data: UserLogin, db: Session = Depends(get_db)):
    # Find user by email
    user = find_user_by_email(user_data.email, db)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "JWT"},
        )
    
    # Check if user is verified
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please verify your email before logging in",
            headers={"WWW-Authenticate": "JWT"},
        )
        
    # Verify password
    if not verify_password(user_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "JWT"},
        )
        
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )
    
    user_response = UserResponse.model_validate(user)
    
    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "JWT",
        "user": user_response
    }

def forgot_password(email_data: PasswordResetRequest, db: Session = Depends(get_db)):
    user = find_user_by_email(email_data.email, db)
    
    # Always return success to prevent email enumeration
    if not user:
        return {
            "message": "If your email is registered, you will receive a password reset link shortly."
        }
    
    # Generate reset token
    reset_token = str(uuid.uuid4())
    token_expires = datetime.now() + timedelta(hours=24)
    
    # Update user with reset token
    user.reset_token = reset_token
    user.token_expires = token_expires
    db.commit()
    
    # Send reset email
    reset_link = f"{FRONTEND_URL}/reset-password?token={reset_token}"
    EmailService.send_reset_password_email(
        to_email=user.email,
        user_name=user.name,
        reset_link=reset_link
    )
    
    return {
        "message": "If your email is registered, you will receive a password reset link shortly."
    }

def reset_password(token: str, password_data: PasswordReset, db: Session = Depends(get_db)):
    # Find user by reset token
    user = find_user_by_token(token, "reset", db)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Check if token is expired
    if user.token_expires < datetime.now():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired"
        )
    
    # Update password
    password_hash = get_password_hash(password_data.password)
    updated_user = update_user_password(user.id, password_hash, db)
    
    return {
        "message": "Password has been successfully reset"
    }