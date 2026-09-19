from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from app.models.user import UserRole


class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Optional[UserRole] = UserRole.OFFICER
    department: Optional[str] = "Legal Metrology"


class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: UserRole
    department: str
    created_at: datetime

    class Config:
        from_attributes = True
