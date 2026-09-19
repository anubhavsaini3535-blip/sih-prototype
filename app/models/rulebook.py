from sqlalchemy import Column, Integer, String, Boolean
from app.database import Base


class RequirementType(str):
    MANDATORY = "mandatory"
    CONDITIONAL = "conditional"
    NOT_APPLICABLE = "not_applicable"


class Rulebook(Base):
    __tablename__ = "rulebooks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category = Column(String(50), nullable=False, index=True)
    field_name = Column(String(100), nullable=False)
    is_mandatory = Column(Boolean, default=True, nullable=False)
    requirement_type = Column(String(20), nullable=False, default="mandatory")
    condition_description = Column(String(300), nullable=True)
    validation_regex = Column(String(500), nullable=True)
