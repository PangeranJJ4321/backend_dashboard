from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from fastapi import Form, UploadFile, File
from uuid import UUID


class UserBase(BaseModel):
    name: str
    email: EmailStr


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    photo: Optional[str] = None  

    @classmethod
    def as_form(
        cls,
        name: str = Form(...),
        email: EmailStr = Form(...),
        password: str = Form(...),
    ):
        # photo is initialized as None and will be set after Cloudinary upload
        return cls(name=name, email=email, password=password, photo=None)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID
    name: str
    email: EmailStr
    photo: str
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None  

    class Config:
        from_attributes = True


class RegisterResponse(BaseModel):
    message: str
    user: UserResponse


class VerifyEmailResponse(BaseModel):
    message: str
    access_token: str
    token_type: str
    user: UserResponse


class LoginResponse(BaseModel):
    message: str
    access_token: str
    token_type: str
    user: UserResponse


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordReset(BaseModel):
    new_password: str
    confirm_password: str
    
    @field_validator('new_password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v
    
    @field_validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class Token(BaseModel):
    access_token: str
    token_type: str


class MessageResponse(BaseModel):
    message: str