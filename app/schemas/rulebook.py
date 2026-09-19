from typing import Optional
from pydantic import BaseModel


class RulebookBase(BaseModel):
    category: str
    field_name: str
    is_mandatory: bool = True
    validation_regex: Optional[str] = None


class RulebookCreate(RulebookBase):
    pass


class RulebookResponse(RulebookBase):
    id: int

    class Config:
        from_attributes = True
