import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.config import (
    CONFIDENCE_PASS_THRESHOLD,
    CONFIDENCE_REVIEW_THRESHOLD,
    OVERALL_OCR_CONFIDENCE_THRESHOLD,
    FONT_SIZE_RATIO_THRESHOLD,
)
from app.models.rulebook import Rulebook
from app.models.scan_result import ComplianceStatus
from app.services.extraction_service import ExtractedField


class EvaluatedRuleResult:
    def __init__(
        self,
        field_name: str,
        requirement_type: str,
        extracted_value: Optional[str],
        compliance_status: ComplianceStatus,
        confidence_score: float,
        bounding_box: Optional[Dict[str, Any]] = None,
        pending_reason: Optional[str] = None,
        match_layer: Optional[str] = None,
        ocr_attempted: bool = True,
    ):
        self.field_name = field_name
        self.requirement_type = requirement_type
        self.extracted_value = extracted_value
        self.compliance_status = compliance_status
        self.confidence_score = round(confidence_score, 4)
        self.bounding_box = bounding_box
        self.pending_reason = pending_reason
        self.match_layer = match_layer
        self.ocr_attempted = ocr_attempted
        # Backward-compat alias
        self.is_mandatory = (requirement_type == "mandatory")
        self.review_reason = pending_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "requirement_type": self.requirement_type,
            "extracted_value": self.extracted_value,
            "compliance_status": self.compliance_status.value,
            "confidence_score": self.confidence_score,
            "bounding_box": self.bounding_box,
            "pending_reason": self.pending_reason,
            "match_layer": self.match_layer,
            "ocr_attempted": self.ocr_attempted,
        }


def check_compliance_rules(
    category: str,
    extracted_fields: Dict[str, ExtractedField],
    overall_ocr_confidence: float,
    median_text_height: float,
    db: Session,
) -> List[EvaluatedRuleResult]:
    """
    Evaluates extracted fields against database Rulebook entries for the given category.

    3-Tag compliance decision logic:
    - NOT_REQUIRED:              not_applicable rules, OR conditional rules where field is absent
    - PENDING_USER_CONFIRMATION: mandatory field not found, OR low-confidence detection
    - COMPLIANT:                 field detected with sufficient confidence + format OK
    - NON_COMPLIANT:             set only by officer via /confirm endpoint (not auto-assigned)

    Font size and format checks that previously caused auto-fail now raise
    PENDING_USER_CONFIRMATION for officer review instead.
    """
    # Fetch rulebook entries for category (fallback to general_retail if not seeded)
    rules = db.query(Rulebook).filter(Rulebook.category == category).all()
    if not rules:
        rules = db.query(Rulebook).filter(Rulebook.category == "general_retail").all()

    evaluated_results: List[EvaluatedRuleResult] = []

    for rule in rules:
        field_name = rule.field_name
        req_type = getattr(rule, "requirement_type", None) or (
            "mandatory" if rule.is_mandatory else "conditional"
        )
        condition_desc = getattr(rule, "condition_description", None)

        # ── NOT_APPLICABLE: immediately exempt, skip all checks ──────────────
        if req_type == "not_applicable":
            evaluated_results.append(
                EvaluatedRuleResult(
                    field_name=field_name,
                    requirement_type=req_type,
                    extracted_value=None,
                    compliance_status=ComplianceStatus.NOT_REQUIRED,
                    confidence_score=0.0,
                    pending_reason="Field is not applicable for this product category.",
                    ocr_attempted=False,
                )
            )
            continue

        # Map database field name to the internal extracted field name
        extraction_key = field_name
        if field_name == "manufacturer_name_address":
            extraction_key = "manufacturer_details"

        extracted = extracted_fields.get(extraction_key)

        if extracted and extracted.value:
            val = extracted.value
            conf = extracted.confidence
            box = extracted.bounding_box
            height = extracted.height
            issues = []

            # Font Size Check disabled — small print (dates, manufacturer, consumer care)
            # is legally permitted on packaging labels. Enabling this check caused valid
            # small-text fields to be incorrectly flagged as pending/non-compliant.
            # Rule 7/8 font-size violations should only be raised by a human inspector.

            # 2. Format Validation (lenient — only flags, doesn't auto-fail)
            if rule.validation_regex:
                try:
                    norm_val = re.sub(r"\s+", " ", val).strip()
                    matched = bool(re.search(rule.validation_regex, norm_val, re.IGNORECASE))

                    # Currency symbol tolerance for MRP (e.g. '₹ 135.00' matching 'Rs.|INR|MRP' pattern)
                    if not matched and field_name == "mrp":
                        mrp_norm = norm_val.replace("₹", "Rs. ")
                        if re.search(rule.validation_regex, mrp_norm, re.IGNORECASE) or re.search(r"\b\d+(?:\.\d{1,2})?\b", norm_val):
                            matched = True

                    # Consumer Care Rule 6(1)(da) tolerance (email or phone or toll-free)
                    elif not matched and field_name == "consumer_care":
                        has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", norm_val))
                        has_phone = bool(re.search(r"(?:\+91|0|\b18[06]0|\b\d{10}\b)", norm_val))
                        if has_email or has_phone:
                            matched = True

                    # Date format tolerance (e.g. '3/27', '03/2027', '15/03/2027', or 'Month YYYY')
                    elif not matched and field_name in ("mfg_date", "best_before_expiry", "expiry_date"):
                        if re.search(r"\b(?:0?[1-9]|[12]\d|3[01])?[\/\-\.](?:0?[1-9]|1[0-2])[\/\-\.](?:20\d{2}|\d{2})\b|\b(?:0?[1-9]|1[0-2])[\/\-\.](?:20\d{2}|\d{2})\b|[A-Za-z]{3,9}[\s\/\-\.](?:20\d{2}|\d{2})", norm_val):
                            matched = True

                    if not matched:
                        issues.append(
                            f"Format mismatch (expected pattern: {rule.validation_regex})"
                        )
                except re.error:
                    pass

            # 3. Confidence-based decision
            if conf >= CONFIDENCE_PASS_THRESHOLD and not issues:
                compliance_status = ComplianceStatus.COMPLIANT
                pending_reason = None
            elif conf >= CONFIDENCE_PASS_THRESHOLD and issues:
                # High-confidence OCR but minor format/font issue → flag for review
                compliance_status = ComplianceStatus.PENDING_USER_CONFIRMATION
                pending_reason = (
                    f"Field detected but requires officer verification: {'; '.join(issues)}"
                )
            elif conf >= CONFIDENCE_REVIEW_THRESHOLD:
                compliance_status = ComplianceStatus.PENDING_USER_CONFIRMATION
                pending_reason = (
                    f"Confidence {conf:.2f} within review range. "
                    + (f"Issues: {'; '.join(issues)}" if issues else "Officer verification recommended.")
                )
            else:
                compliance_status = ComplianceStatus.PENDING_USER_CONFIRMATION
                pending_reason = (
                    f"Low OCR confidence ({conf:.2f}). "
                    + (f"Issues: {'; '.join(issues)}" if issues else "Manual verification required.")
                )

            evaluated_results.append(
                EvaluatedRuleResult(
                    field_name=field_name,
                    requirement_type=req_type,
                    extracted_value=val,
                    compliance_status=compliance_status,
                    confidence_score=conf,
                    bounding_box=box,
                    pending_reason=pending_reason,
                    match_layer=extracted.match_layer if extracted else None,
                )
            )

        else:
            # Field not detected by OCR
            if req_type == "mandatory":
                # Mandatory missing → always PENDING_USER_CONFIRMATION (never auto-fail)
                if overall_ocr_confidence >= OVERALL_OCR_CONFIDENCE_THRESHOLD:
                    pending_reason = (
                        f"Mandatory declaration '{field_name}' not found on label "
                        f"(OCR confidence: {overall_ocr_confidence:.2f}). "
                        f"Officer: please visually verify this packaging."
                    )
                else:
                    pending_reason = (
                        f"Mandatory declaration '{field_name}' not found "
                        f"(low overall OCR clarity: {overall_ocr_confidence:.2f}). "
                        f"Retake photo or manually confirm."
                    )
                evaluated_results.append(
                    EvaluatedRuleResult(
                        field_name=field_name,
                        requirement_type=req_type,
                        extracted_value=None,
                        compliance_status=ComplianceStatus.PENDING_USER_CONFIRMATION,
                        confidence_score=0.0,
                        pending_reason=pending_reason,
                        ocr_attempted=overall_ocr_confidence > 0,
                    )
                )
            else:
                # Conditional field absent → assume condition doesn't apply → NOT_REQUIRED
                not_req_reason = (
                    f"Conditional field '{field_name}' not detected. "
                    + (f"Condition: {condition_desc}. " if condition_desc else "")
                    + "Assumed not applicable. Officer can override via /confirm if required."
                )
                evaluated_results.append(
                    EvaluatedRuleResult(
                        field_name=field_name,
                        requirement_type=req_type,
                        extracted_value=None,
                        compliance_status=ComplianceStatus.NOT_REQUIRED,
                        confidence_score=0.0,
                        pending_reason=not_req_reason,
                        ocr_attempted=True,
                    )
                )

    return evaluated_results
