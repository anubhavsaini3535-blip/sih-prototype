from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class OcrCorrection(Base):
    """
    Audit log for officer confirmations on OCR-failed or low-confidence fields.
    Stored for future model improvement and traceability under Legal Metrology Act.
    """
    __tablename__ = "ocr_corrections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey("scans.id"), nullable=False, index=True)
    result_id = Column(Integer, ForeignKey("scan_results.id"), nullable=False, index=True)
    field_name = Column(String(100), nullable=False)
    officer_decision = Column(String(50), nullable=False)   # field_is_present | not_applicable_override
    override_reason = Column(String(500), nullable=True)    # officer-supplied free text
    image_region = Column(JSON, nullable=True)              # bounding box of region in question
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    scan = relationship("Scan")
