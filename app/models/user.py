import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.orm import relationship
from app.database import Base


class UserRole(str, enum.Enum):
    OFFICER = "officer"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole, values_callable=lambda obj: [e.value for e in obj]), default=UserRole.OFFICER, nullable=False)
    department = Column(String(150), nullable=False, default="Legal Metrology")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    scans = relationship("Scan", back_populates="officer", cascade="all, delete-orphan")
