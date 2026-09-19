import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class SourceType(str, enum.Enum):
    PHYSICAL_LABEL = "physical_label"
    ECOMMERCE_LISTING = "ecommerce_listing"


class ProductCategory(str, enum.Enum):
    GENERAL_RETAIL = "general_retail"
    FOOD = "food"
    FOOD_BEVERAGES = "food_beverages"
    COSMETICS = "cosmetics"
    ELECTRONICS = "electronics"
    MEDICINE = "medicine"
    CLOTHES = "clothes"
    STATIONERY = "stationery"


class ScanStatus(str, enum.Enum):
    PROCESSING = "processing"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    officer_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    image_url = Column(String(500), nullable=False)
    source_type = Column(
        Enum(SourceType, values_callable=lambda obj: [e.value for e in obj]),
        default=SourceType.PHYSICAL_LABEL,
        nullable=False,
    )
    product_category = Column(
        Enum(ProductCategory, values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    status = Column(
        Enum(ScanStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=ScanStatus.PROCESSING,
        nullable=False,
    )
    location = Column(String(255), nullable=True)
    shop_name = Column(String(255), nullable=True)
    final_decision = Column(String(50), nullable=True)  # "compliant" or "non_compliant"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    officer = relationship("User", back_populates="scans")
    results = relationship("ScanResult", back_populates="scan", cascade="all, delete-orphan")

    @property
    def image_urls(self) -> list:
        """Returns list of image URLs (supporting up to 3 multi-panel packaging photos)."""
        if not self.image_url:
            return []
        return [u.strip() for u in self.image_url.split(",") if u.strip()]

