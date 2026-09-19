from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    require_admin,
)
from app.services.image_service import (
    check_image_quality,
    preprocess_image,
)
from app.services.ocr_service import run_ocr
from app.services.extraction_service import (
    extract_from_blocks,
    extract_from_text,
)
from app.services.rule_checker import check_compliance_rules
from app.services.scraper_service import scrape_ecommerce_page
from app.services.report_service import generate_pdf_report, generate_docx_report
from app.services.field_matcher import (
    field_matcher,
    FieldMatcher,
    FieldMatchResult,
)
from app.services.gemini_service import (
    get_ai_resolved_matches,
    log_ai_resolved_match,
)

__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "get_current_user",
    "require_admin",
    "check_image_quality",
    "preprocess_image",
    "run_ocr",
    "extract_from_blocks",
    "extract_from_text",
    "check_compliance_rules",
    "scrape_ecommerce_page",
    "generate_pdf_report",
    "generate_docx_report",
    "field_matcher",
    "FieldMatcher",
    "FieldMatchResult",
    "get_ai_resolved_matches",
    "log_ai_resolved_match",
]
