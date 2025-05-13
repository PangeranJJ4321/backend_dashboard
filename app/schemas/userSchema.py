from pydantic import BaseModel, EmailStr, UUID4, validator
from typing import Optional, List
from datetime import datetime
from fastapi import Form

class UserBase(BaseModel):
    name: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    photo: Optional[str] = None
    
    @classmethod
    def as_form(
        cls,
        name: Optional[str] = Form(None),
        email: Optional[EmailStr] = Form(None),
    ):
        # photo is initialized as None and will be set after Cloudinary upload
        return cls(name=name, email=email, photo=None)

class UserChangePassword(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str
    
    @validator('new_password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

class UserResponse(UserBase):
    id: UUID4
    photo: Optional[str] = None
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ProfileUpdateResponse(BaseModel):
    message: str
    user: UserResponse
    
class MessageResponse(BaseModel):
    message: str