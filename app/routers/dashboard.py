from datetime import datetime, timedelta
from typing import Dict, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.database import get_db
from app.models.user import User
from app.models.scan import Scan, ScanStatus, ProductCategory
from app.models.scan_result import ScanResult, ComplianceStatus, OfficerOverride
from app.schemas.dashboard import DashboardStatsResponse, DailyTrend, MissingFieldStat
from app.services.auth_service import require_admin

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Admin-only aggregate statistics.
    Violation rate is calculated using only NON_COMPLIANT declarations.
    NOT_REQUIRED fields are excluded from the denominator so categories with
    many not-applicable fields are not skewed artificially.
    """
    total_scans = db.query(Scan).count()

    # Scans with at least one NON_COMPLIANT or officer-confirmed missing
    violating_scan_ids = (
        db.query(distinct(ScanResult.scan_id))
        .filter(
            (ScanResult.compliance_status == ComplianceStatus.NON_COMPLIANT)
            | (ScanResult.officer_override == OfficerOverride.CONFIRM_MISSING)
        )
        .all()
    )
    violating_scan_id_set = {r[0] for r in violating_scan_ids}
    total_violations = len(violating_scan_id_set)

    overall_violation_rate = (
        round((total_violations / total_scans) * 100, 2) if total_scans > 0 else 0.0
    )

    # Category-wise violation rates
    category_violation_rates: Dict[str, float] = {}
    for cat in ProductCategory:
        cat_scans = db.query(Scan).filter(Scan.product_category == cat).all()
        cat_total = len(cat_scans)
        if cat_total == 0:
            category_violation_rates[cat.value] = 0.0
        else:
            cat_viols = sum(1 for s in cat_scans if s.id in violating_scan_id_set)
            category_violation_rates[cat.value] = round((cat_viols / cat_total) * 100, 2)

    # Scan status breakdown (now includes awaiting_review)
    status_breakdown: Dict[str, int] = {}
    for st in ScanStatus:
        cnt = db.query(Scan).filter(Scan.status == st).count()
        status_breakdown[st.value] = cnt

    # Compliance breakdown (4 states; NOT_REQUIRED excluded from violation denominator)
    compliance_breakdown: Dict[str, int] = {}
    for cs in ComplianceStatus:
        cnt = db.query(ScanResult).filter(ScanResult.compliance_status == cs).count()
        compliance_breakdown[cs.value] = cnt

    # Per-category field-level violation denominator (excluding NOT_REQUIRED)
    compliant_count = compliance_breakdown.get("compliant", 0)
    non_compliant_count = compliance_breakdown.get("non_compliant", 0)
    pending_count = compliance_breakdown.get("pending_user_confirmation", 0)
    evaluated_declarations = compliant_count + non_compliant_count + pending_count
    field_violation_rate = (
        round((non_compliant_count / evaluated_declarations) * 100, 2)
        if evaluated_declarations > 0 else 0.0
    )

    # Trend over the last 30 days
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=29)
    daily_trends: List[DailyTrend] = []
    for day_offset in range(30):
        current_day = start_date + timedelta(days=day_offset)
        next_day = current_day + timedelta(days=1)
        day_str = current_day.strftime("%Y-%m-%d")
        day_scans = (
            db.query(Scan)
            .filter(
                Scan.created_at >= datetime.combine(current_day, datetime.min.time()),
                Scan.created_at < datetime.combine(next_day, datetime.min.time()),
            )
            .all()
        )
        day_total = len(day_scans)
        day_viols = sum(1 for s in day_scans if s.id in violating_scan_id_set)
        daily_trends.append(
            DailyTrend(date=day_str, total_scans=day_total, violations=day_viols)
        )

    # Most commonly missing / failing mandatory declarations
    # Excludes NOT_REQUIRED from the missing-field query
    missing_query = (
        db.query(
            ScanResult.field_name,
            func.count(ScanResult.id).label("count"),
        )
        .filter(
            (ScanResult.compliance_status == ComplianceStatus.NON_COMPLIANT)
            | (
                (ScanResult.compliance_status == ComplianceStatus.PENDING_USER_CONFIRMATION)
                & (ScanResult.extracted_value.is_(None))
            )
        )
        .group_by(ScanResult.field_name)
        .order_by(func.count(ScanResult.id).desc())
        .limit(10)
        .all()
    )
    most_commonly_missing: List[MissingFieldStat] = []
    for item in missing_query:
        field_name, cnt = item[0], item[1]
        pct = round((cnt / total_scans) * 100, 2) if total_scans > 0 else 0.0
        most_commonly_missing.append(
            MissingFieldStat(field_name=field_name, count=cnt, percentage=pct)
        )

    return DashboardStatsResponse(
        total_scans=total_scans,
        total_violations=total_violations,
        overall_violation_rate_pct=overall_violation_rate,
        category_violation_rates=category_violation_rates,
        status_breakdown=status_breakdown,
        compliance_breakdown=compliance_breakdown,
        field_violation_rate_pct=field_violation_rate,
        last_30_days_trend=daily_trends,
        most_commonly_missing_fields=most_commonly_missing,
    )


@router.get("/ai-matches")
def list_ai_resolved_matches(
    limit: int = 50,
    admin_user: User = Depends(require_admin),
):
    """
    Admin-only review of Layer-3 AI-resolved matches logged from Gemini API.
    """
    from app.config import AI_MATCH_LOG_FILE
    from app.services.gemini_service import get_ai_resolved_matches

    matches = get_ai_resolved_matches(limit=limit)
    return {
        "matches": matches,
        "count": len(matches),
        "model": "gemini-flash-latest",
        "log_path": str(AI_MATCH_LOG_FILE),
    }


@router.get("/ocr-corrections")
def list_ocr_corrections(
    limit: int = 50,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Admin-only: view all officer confirmations logged for model-improvement review.
    Covers both 'field_is_present' (OCR miss) and 'not_applicable_override' decisions.
    """
    from app.models.ocr_correction import OcrCorrection
    corrections = (
        db.query(OcrCorrection)
        .order_by(OcrCorrection.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "corrections": [
            {
                "id": c.id,
                "scan_id": c.scan_id,
                "result_id": c.result_id,
                "field_name": c.field_name,
                "officer_decision": c.officer_decision,
                "override_reason": c.override_reason,
                "image_region": c.image_region,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in corrections
        ],
        "count": len(corrections),
    }
