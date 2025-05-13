from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import Optional
from app.core.session import get_db
from app.middleware.security import get_current_user
from app.models.models import User
from app.schemas.userSchema import UserResponse, UserUpdate, ProfileUpdateResponse, UserChangePassword, MessageResponse
from app.controllers.userController import UserController

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.post("/me/change-password", response_model=MessageResponse)
async def change_password(
    password_data: UserChangePassword,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user password
    """
    try:
        return UserController.change_user_password(db, current_user.id, password_data, current_user)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )

@router.get("/me", response_model=UserResponse)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user profile information
    """
    return UserController.get_current_user_profile(db, current_user)

@router.put("/me", response_model=ProfileUpdateResponse)
async def update_user_profile(
    user_data: UserUpdate = Depends(UserUpdate.as_form),
    photo_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current user profile information
    """
    try:
        updated_user = UserController.update_user_profile(db, current_user.id, user_data, photo_file)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return {
            "message": "Profile updated successfully",
            "user": updated_user
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile update failed: {str(e)}"
        )