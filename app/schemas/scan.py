from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel
from app.models.scan import SourceType, ProductCategory, ScanStatus
from app.models.scan_result import ComplianceStatus, OfficerOverride
from app.schemas.auth import UserResponse


class ScanResultResponse(BaseModel):
    id: int
    scan_id: int
    field_name: str
    extracted_value: Optional[str] = None
    compliance_status: ComplianceStatus
    confidence_score: float
    bounding_box: Optional[object] = None
    officer_override: Optional[OfficerOverride] = None
    match_layer: Optional[str] = None
    # Multi-category + OCR-confirmation fields
    requirement_type: Optional[str] = "mandatory"
    ocr_attempted: bool = True
    pending_reason: Optional[str] = None

    # Human-readable display tag for reports / UI
    @property
    def display_status(self) -> str:
        if self.compliance_status == ComplianceStatus.COMPLIANT:
            return "Compliant"
        if self.compliance_status == ComplianceStatus.NON_COMPLIANT:
            return "Non-Compliant"
        if self.compliance_status == ComplianceStatus.NOT_REQUIRED:
            return "Not Required"
        return "Awaiting Officer Review"

    class Config:
        from_attributes = True


class ScanResponse(BaseModel):
    id: int
    officer_id: int
    image_url: str
    image_urls: List[str] = []
    source_type: SourceType
    product_category: ProductCategory
    status: ScanStatus
    location: Optional[str] = None
    shop_name: Optional[str] = None
    final_decision: Optional[str] = None
    created_at: datetime
    results: List[ScanResultResponse] = []
    officer: Optional[UserResponse] = None
    # Summary counts for quick UI rendering
    pending_confirmations: int = 0

    class Config:
        from_attributes = True

    @classmethod
    def from_orm_with_counts(cls, scan):
        obj = cls.model_validate(scan)
        obj.pending_confirmations = sum(
            1 for r in obj.results
            if r.compliance_status == ComplianceStatus.PENDING_USER_CONFIRMATION
        )
        return obj


class ScanUrlRequest(BaseModel):
    url: str
    product_category: ProductCategory
    location: Optional[str] = None


class OfficerReviewRequest(BaseModel):
    """Legacy override endpoint (kept for backward compat)."""
    officer_override: OfficerOverride


class OfficerConfirmRequest(BaseModel):
    """New confirmation endpoint: officer resolves a pending_user_confirmation field."""
    user_decision: Literal["confirm_missing", "field_is_present", "not_applicable_override"]
    override_reason: Optional[str] = None


class ScanFinalizeRequest(BaseModel):
    shop_name: Optional[str] = None
    final_decision: Literal["compliant", "non_compliant"]


class ScanListResponse(BaseModel):
    items: List[ScanResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
