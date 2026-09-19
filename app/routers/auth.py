from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.schemas.auth import UserRegister, UserLogin, Token, UserResponse, ForgotPasswordRequest
from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserRegister, db: Session = Depends(get_db)):
    """Registers a new officer or admin account."""
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    new_user = User(
        name=user_in.name,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        role=user_in.role,
        department=user_in.department or "Legal Metrology",
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Authenticates user and returns JWT token.
    Works directly with Swagger UI OAuth2 password flow (form_data.username is treated as email).
    """
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role.value}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login/json", response_model=Token)
def login_json(
    credentials: UserLogin,
    db: Session = Depends(get_db),
):
    """Convenience JSON login endpoint for programmatic API clients."""
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role.value}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Returns profile information for the authenticated user."""
    return current_user

@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Mock forgot password endpoint for prototype."""
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    
    # Auto-reset password for prototype purposes
    temp_password = "reset123"
    user.hashed_password = hash_password(temp_password)
    db.commit()
    
    return {"message": f"Password has been reset to: {temp_password}"}

