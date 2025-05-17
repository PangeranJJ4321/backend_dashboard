from datetime import datetime, timedelta
import os
import uuid
from dotenv import load_dotenv
from app.middleware.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES, 
    create_access_token, 
    get_password_hash, 
    verify_password
)
from app.repositories.authRepositories import (
    create_new_user, 
    find_user_by_email, 
    find_user_by_token, 
    update_user_password, 
    update_user_verification
)
from app.schemas.authSchema import (
    PasswordReset, 
    PasswordResetRequest, 
    UserCreate, 
    UserLogin
)
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.services.email_service import EmailService


# Load environment variables
load_dotenv()
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

def register_user(user_data: UserCreate, db: Session):
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

    return new_user

def verify_email_user(token : str, db: Session):
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

    # Send welcome email
    EmailService.send_welcome_email(
        to_email=verified_user.email,
        user_name=verified_user.name
    )

    return verified_user, access_token


def login_user(user_data: UserLogin, db: Session):
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
            headers={"WWW-Authenticate": "bearer"},
        )
        
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )

    return user, access_token

def forgot_password_user(email_data: PasswordResetRequest, db: Session):
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

    return True


def reset_password_user(token: str, password_data: PasswordReset, db: Session):
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
    
    # Check if passwords match
    if password_data.new_password != password_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # Update password
    password_hash = get_password_hash(password_data.new_password)
    update_user_password(user.id, password_hash, db)

    return True

    