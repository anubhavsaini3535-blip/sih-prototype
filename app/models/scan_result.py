import enum
from sqlalchemy import Column, Integer, String, Float, Boolean, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class ComplianceStatus(str, enum.Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_REQUIRED = "not_required"
    PENDING_USER_CONFIRMATION = "pending_user_confirmation"


class OfficerOverride(str, enum.Enum):
    CONFIRM_MISSING = "confirm_missing"
    FIELD_IS_PRESENT = "field_is_present"
    NOT_APPLICABLE_OVERRIDE = "not_applicable_override"


class ScanResult(Base):
    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey("scans.id"), nullable=False, index=True)
    field_name = Column(String(100), nullable=False, index=True)
    extracted_value = Column(String(500), nullable=True)
    compliance_status = Column(
        Enum(ComplianceStatus, values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    confidence_score = Column(Float, nullable=False, default=0.0)
    bounding_box = Column(JSON, nullable=True)
    officer_override = Column(
        Enum(OfficerOverride, values_callable=lambda obj: [e.value for e in obj]),
        nullable=True,
    )
    match_layer = Column(String(50), nullable=True)
    # New columns for multi-category requirement awareness + OCR confirmation flow
    requirement_type = Column(String(20), nullable=True, default="mandatory")
    ocr_attempted = Column(Boolean, nullable=False, default=True)
    pending_reason = Column(String(500), nullable=True)

    scan = relationship("Scan", back_populates="results")
