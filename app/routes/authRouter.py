from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Path, Query
from sqlalchemy.orm import Session

from app.core.session import get_db
from app.controllers.authController import (
    register, 
    login, 
    verify_email, 
    forgot_password, 
    reset_password
)
from app.schemas.schemaUser import (
    RegisterResponse, 
    UserCreate, 
    UserResponse, 
    UserLogin, 
    LoginResponse,
    VerifyEmailResponse,
    PasswordResetRequest,
    PasswordReset,
    MessageResponse
)
from app.utils.couldinary import upload_image_to_cloudinary

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=RegisterResponse)
async def user_register(
    user_data: UserCreate = Depends(UserCreate.as_form),
    photo_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Upload to Cloudinary
        photo_url = upload_image_to_cloudinary(photo_file)

        # Modify user_data to include photo URL
        user_data_dict = user_data.dict()
        user_data_dict['photo'] = photo_url

        return register(UserCreate(**user_data_dict), db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.get("/verify-email", response_model=VerifyEmailResponse)
async def verify_user_email(
    token: str = Query(..., description="Email verification token"),
    db: Session = Depends(get_db)
):
    try:
        return verify_email(token, db)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email verification failed: {str(e)}"
        )

@router.post("/login", response_model=LoginResponse)
async def user_login(
    user_login: UserLogin,
    db: Session = Depends(get_db)
):
    try:
        return login(user_login, db)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post("/forgot-password", response_model=MessageResponse)
async def request_password_reset(
    password_reset: PasswordResetRequest,
    db: Session = Depends(get_db)
):
    try:
        return forgot_password(password_reset, db)
    except Exception as e:
        # Always return same message even if error occurs to prevent email enumeration
        return {"message": "If your email is registered, you will receive a password reset link shortly."}

@router.post("/reset-password", response_model=MessageResponse)
async def reset_user_password(
    token: str = Query(..., description="Password reset token"),
    password_data: PasswordReset = Depends(),
    db: Session = Depends(get_db)
):
    try:
        return reset_password(token, password_data, db)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset failed: {str(e)}"
        )