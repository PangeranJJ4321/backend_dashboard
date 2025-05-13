from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import UploadFile, HTTPException, status
from app.models.models import User
from app.schemas.userSchema import UserUpdate, UserResponse, UserChangePassword
from app.repositories.userRepositories import UserRepository
from app.utils.couldinary import upload_image_to_cloudinary
from app.middleware.security import verify_password, get_password_hash
from sqlalchemy.orm import Session 

class UserController:
    @staticmethod
    def get_current_user_profile(db: Session, current_user: User) -> UserResponse:
        """
        Get current user profile information
        """
        return current_user


    @staticmethod
    def update_user_profile(
        db: Session, 
        user_id: UUID, 
        user_data: UserUpdate, 
        photo_file: Optional[UploadFile] = None
    ) -> Optional[User]:
        """
        Update current user profile information
        """
        # If a new photo is uploaded, process it
        if photo_file:
            try:
                # Upload to Cloudinary
                photo_url = upload_image_to_cloudinary(photo_file)
                
                # Update the user_data with the new photo URL
                user_data_dict = user_data.dict(exclude_unset=True)
                user_data_dict['photo'] = photo_url
                
                # Create a new UserUpdate with the photo URL
                user_data = UserUpdate(**user_data_dict)
            except Exception as e:
                # Handle upload errors
                raise e
                
        return UserRepository.update_user(db, user_id, user_data)
    
    @staticmethod
    def change_user_password(
        db: Session, 
        user_id: UUID,
        password_data: UserChangePassword,
        current_user: User
    ) -> Dict[str, str]:
        """
        Change user password
        """
        # Verify current password
        if not verify_password(password_data.current_password, current_user.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
            
        # Check if passwords match
        if password_data.new_password != password_data.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Passwords do not match"
            )
        
        # Generate new password hash
        hashed_password = get_password_hash(password_data.new_password)
        
        # Update password
        UserRepository.update_user_password(db, user_id, hashed_password)
        
        return {"message": "Password changed successfully"}