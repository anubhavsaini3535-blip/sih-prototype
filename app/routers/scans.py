import os
import uuid
import shutil
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, joinedload

import numpy as np
from app.config import UPLOAD_DIR, BLUR_THRESHOLD, MAX_SCAN_IMAGES
from app.database import get_db
from app.models.user import User, UserRole
from app.models.scan import Scan, SourceType, ProductCategory, ScanStatus
from app.models.scan_result import ScanResult, ComplianceStatus, OfficerOverride
from app.models.ocr_correction import OcrCorrection
from app.schemas.scan import (
    ScanResponse,
    ScanResultResponse,
    ScanUrlRequest,
    OfficerReviewRequest,
    OfficerConfirmRequest,
    ScanFinalizeRequest,
    ScanListResponse,
)
from app.services.auth_service import get_current_user
from app.services.image_service import check_image_quality, preprocess_image
from app.services.ocr_service import run_ocr
from app.services.extraction_service import extract_from_blocks, extract_from_text
from app.services.rule_checker import check_compliance_rules
from app.services.scraper_service import scrape_ecommerce_page
from app.services.report_service import generate_pdf_report, generate_docx_report

router = APIRouter(prefix="/scans", tags=["Scans"])


@router.post("/upload", response_model=ScanResponse, status_code=status.HTTP_201_CREATED)
async def upload_label_scan(
    files: List[UploadFile] = File(default=[], description="One or more photographs of packaging panels (max 3 images: front, back, sides)"),
    file: Optional[UploadFile] = File(None, description="Single photograph of packaged commodity label (backward compatibility)"),
    product_category: ProductCategory = Form(..., description="Commodity category"),
    location: Optional[str] = Form(None, description="Physical retail inspection location"),
    shop_name: Optional[str] = Form(None, description="Name of the retail shop"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Uploads up to 3 packaged product label photos (e.g. front panel, back panel, side panel)
    and executes the synchronous Legal Metrology compliance pipeline across all submitted panels:
    1. Multi-image Quality Check (Laplacian Blur Detection)
    2. Preprocessing (Auto-orientation + Deskewing + Contrast Enhancement)
    3. PaddleOCR local inference per image panel
    4. Mandatory Rule 6 field extraction across combined image blocks (MRP, Net Qty, Mfg Date, Mfg Details, etc.)
    5. Rule compliance checking & relative font-size anomaly evaluation
    """
    # Collect all submitted files
    upload_files: List[UploadFile] = []
    if files:
        upload_files.extend(files)
    if file and file not in upload_files:
        upload_files.append(file)

    # Filter out empty entries
    upload_files = [f for f in upload_files if f and f.filename]
    if not upload_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one packaging label image file must be uploaded.",
        )
    if len(upload_files) > MAX_SCAN_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {MAX_SCAN_IMAGES} images allowed per packaging inspection scan. Received {len(upload_files)} images.",
        )

    # Validate file extensions & save uploaded files to disk
    saved_paths = []
    saved_filenames = []
    for idx, uf in enumerate(upload_files):
        ext = os.path.splitext(uf.filename or "")[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            for sp in saved_paths:
                try:
                    os.remove(sp)
                except OSError:
                    pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image format in file #{idx+1} ('{uf.filename}'). Allowed formats: .jpg, .jpeg, .png, .bmp, .webp",
            )

        unique_filename = f"{uuid.uuid4().hex}{ext}"
        saved_path = str(UPLOAD_DIR / unique_filename)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(uf.file, buffer)
        saved_paths.append(saved_path)
        saved_filenames.append(unique_filename)

    # Step 1: Image Quality / Blur Check for each submitted panel
    for idx, sp in enumerate(saved_paths):
        is_sharp, variance = check_image_quality(sp, threshold=BLUR_THRESHOLD)
        if not is_sharp:
            for p in saved_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Image #{idx+1} ('{upload_files[idx].filename}') rejected: image is too blurry "
                    f"(Laplacian variance: {variance:.1f} < threshold: {BLUR_THRESHOLD}). "
                    f"Please retake a sharper, focused photograph of this packaging panel."
                ),
            )

    # Create initial scan record storing all image URLs
    image_urls_list = [f"/uploads/{fn}" for fn in saved_filenames]
    scan = Scan(
        officer_id=current_user.id,
        image_url=",".join(image_urls_list),
        source_type=SourceType.PHYSICAL_LABEL,
        product_category=product_category,
        status=ScanStatus.PROCESSING,
        location=location,
        shop_name=shop_name,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    try:
        # Step 2 & 3: Preprocess and Run OCR on each packaging panel
        all_blocks = []
        confidences = []
        heights = []

        for img_idx, (sp, fn) in enumerate(zip(saved_paths, saved_filenames)):
            prep_path = str(UPLOAD_DIR / f"prep_{fn}")
            preprocess_image(sp, prep_path)
            ocr_result = run_ocr(prep_path, image_index=img_idx)
            all_blocks.extend(ocr_result["blocks"])
            confidences.append(ocr_result["average_confidence"])
            heights.append(ocr_result["median_height"])

        avg_overall_conf = float(np.mean(confidences)) if confidences else 0.85
        median_overall_height = float(np.median(heights)) if heights else 18.0

        # Step 4: Field Extraction across unified packaging text blocks
        extracted_fields = extract_from_blocks(all_blocks, scan_id=scan.id)


        # Step 5 & 6: Rulebook Matching & Font Size Check
        evaluated_rules = check_compliance_rules(
            category=product_category.value,
            extracted_fields=extracted_fields,
            overall_ocr_confidence=avg_overall_conf,
            median_text_height=median_overall_height,
            db=db,
        )

        # Save Scan Results
        has_pending = False
        for item in evaluated_rules:
            scan_res = ScanResult(
                scan_id=scan.id,
                field_name=item.field_name,
                extracted_value=item.extracted_value,
                compliance_status=item.compliance_status,
                confidence_score=item.confidence_score,
                bounding_box=item.bounding_box,
                match_layer=item.match_layer,
                requirement_type=item.requirement_type,
                ocr_attempted=item.ocr_attempted,
                pending_reason=item.pending_reason,
            )
            db.add(scan_res)
            if item.compliance_status == ComplianceStatus.PENDING_USER_CONFIRMATION:
                has_pending = True

        scan.status = ScanStatus.AWAITING_REVIEW if has_pending else ScanStatus.COMPLETED
        db.commit()
        db.refresh(scan)

    except Exception as e:
        scan.status = ScanStatus.FAILED
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during pipeline execution: {str(e)}",
        )

    # Return scan with loaded relationships
    return (
        db.query(Scan)
        .options(joinedload(Scan.results), joinedload(Scan.officer))
        .filter(Scan.id == scan.id)
        .first()
    )


@router.post("/{scan_id}/gemini-check", response_model=ScanResponse)
def run_gemini_check(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    On-demand Gemini Vision analysis.
    Reads the first uploaded image for the scan and asks Gemini to fill in
    any fields that are still missing or marked as pending_user_confirmation.
    Updates the ScanResult rows in place and returns the updated scan.
    """
    from app.services.gemini_service import batch_vision_extract_fields, is_ai_available
    from app.services.extraction_service import ExtractedField

    scan = (
        db.query(Scan)
        .options(joinedload(Scan.results), joinedload(Scan.officer))
        .filter(Scan.id == scan_id)
        .first()
    )
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")

    if not is_ai_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API key is not configured. Please add GEMINI_API_KEY to your .env file.",
        )

    # Only works for physical label scans (need an image file)
    if scan.source_type != SourceType.PHYSICAL_LABEL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini Vision check is only available for physical label scans.",
        )

    # Target fields that are pending confirmation, non-compliant, missing, or low confidence
    target_results = {
        r.field_name: r for r in scan.results
        if r.compliance_status in [ComplianceStatus.PENDING_USER_CONFIRMATION, ComplianceStatus.NON_COMPLIANT, "pending_user_confirmation", "non_compliant"]
        or r.extracted_value is None
        or (r.confidence_score is not None and r.confidence_score < 0.85)
    }
    if not target_results:
        # If all fields already have status, allow cross-checking all fields
        target_results = {r.field_name: r for r in scan.results}

    # Also map DB field name -> internal field name
    missing_fields = []
    for db_field_name in target_results:
        internal = "manufacturer_details" if db_field_name == "manufacturer_name_address" else db_field_name
        if internal not in missing_fields:
            missing_fields.append(internal)

    if not missing_fields:
        return (
            db.query(Scan)
            .options(joinedload(Scan.results), joinedload(Scan.officer))
            .filter(Scan.id == scan.id)
            .first()
        )

    # Inspect all uploaded images for target fields
    image_urls = scan.image_urls
    if not image_urls:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No uploaded image found for this scan.")

    vision_results = {}
    remaining_missing = list(missing_fields)
    for img_url in image_urls:
        if not remaining_missing:
            break
        img_filename = os.path.basename(img_url)
        img_path = str(UPLOAD_DIR / img_filename)
        if os.path.exists(img_path):
            panel_results = batch_vision_extract_fields(img_path, remaining_missing)
            for k, v in panel_results.items():
                if v and k not in vision_results:
                    vision_results[k] = v
            remaining_missing = [f for f in missing_fields if f not in vision_results]

    # Update matching ScanResult rows
    updated = 0
    for db_field_name, result_row in target_results.items():
        internal = "manufacturer_details" if db_field_name == "manufacturer_name_address" else db_field_name
        vis_text = vision_results.get(internal)
        if vis_text:
            result_row.extracted_value = vis_text
            result_row.confidence_score = 0.95
            result_row.match_layer = "gemini_vision"
            result_row.compliance_status = ComplianceStatus.COMPLIANT
            result_row.pending_reason = "Verified & extracted via Gemini Vision AI."
            updated += 1

    # Re-evaluate scan completion status
    still_pending = any(
        r.compliance_status in [ComplianceStatus.PENDING_USER_CONFIRMATION, "pending_user_confirmation"] for r in scan.results
    )
    if not still_pending:
        scan.status = ScanStatus.COMPLETED

    db.commit()

    return (
        db.query(Scan)
        .options(joinedload(Scan.results), joinedload(Scan.officer))
        .filter(Scan.id == scan.id)
        .first()
    )


@router.post("/from-url", response_model=ScanResponse, status_code=status.HTTP_201_CREATED)
def scan_from_ecommerce_url(
    payload: ScanUrlRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Scrapes an e-commerce product listing URL and verifies Legal Metrology compliance:
    - Scrapes specifications, description, and title using BeautifulSoup
    - Extracts mandatory declarations (MRP, Net Qty, Manufacturer/Packer/Importer, Consumer Care, Origin)
    - Verifies against ecommerce_listing rulebook (mfg_date is excluded per e-commerce rules)
    """
    # 1. Create Scan record
    scan = Scan(
        officer_id=current_user.id,
        image_url=payload.url,
        source_type=SourceType.ECOMMERCE_LISTING,
        product_category=payload.product_category,
        status=ScanStatus.PROCESSING,
        location=payload.location or "E-Commerce Web Portal",
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    # 2. Scrape listing text
    scraped_text = scrape_ecommerce_page(payload.url)

    # 3. Extract fields from scraped text (3-Layer)
    extracted_fields = extract_from_text(scraped_text, scan_id=scan.id)

    try:
        # Rule check against 'ecommerce_listing' category
        evaluated_rules = check_compliance_rules(
            category="ecommerce_listing",
            extracted_fields=extracted_fields,
            overall_ocr_confidence=0.95,  # Direct text scraping has high fidelity
            median_text_height=16.0,
            db=db,
        )

        has_pending = False
        for item in evaluated_rules:
            scan_res = ScanResult(
                scan_id=scan.id,
                field_name=item.field_name,
                extracted_value=item.extracted_value,
                compliance_status=item.compliance_status,
                confidence_score=item.confidence_score,
                bounding_box=None,
                match_layer=item.match_layer,
                requirement_type=item.requirement_type,
                ocr_attempted=item.ocr_attempted,
                pending_reason=item.pending_reason,
            )
            db.add(scan_res)
            if item.compliance_status == ComplianceStatus.PENDING_USER_CONFIRMATION:
                has_pending = True

        scan.status = ScanStatus.AWAITING_REVIEW if has_pending else ScanStatus.COMPLETED
        db.commit()
        db.refresh(scan)

    except Exception as e:
        scan.status = ScanStatus.FAILED
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process e-commerce listing: {str(e)}",
        )

    return (
        db.query(Scan)
        .options(joinedload(Scan.results), joinedload(Scan.officer))
        .filter(Scan.id == scan.id)
        .first()
    )


@router.get("/history", response_model=ScanListResponse)
def get_scan_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    category: Optional[ProductCategory] = Query(None, description="Filter by product category"),
    compliance_status: Optional[ComplianceStatus] = Query(None, description="Filter scans containing this status"),
    start_date: Optional[datetime] = Query(None, description="Start date filter (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date filter (ISO format)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns a paginated list of scans for the logged-in officer (or all scans if admin).
    Supports category, compliance status, and date range filters.
    """
    query = db.query(Scan).options(joinedload(Scan.results), joinedload(Scan.officer))

    # Officers view their own scans; admins can view all scans
    if current_user.role != UserRole.ADMIN:
        query = query.filter(Scan.officer_id == current_user.id)

    if category:
        query = query.filter(Scan.product_category == category)

    if start_date:
        query = query.filter(Scan.created_at >= start_date)

    if end_date:
        query = query.filter(Scan.created_at <= end_date)

    if compliance_status:
        query = query.join(Scan.results).filter(ScanResult.compliance_status == compliance_status).distinct()

    total = query.count()
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    items = (
        query.order_by(Scan.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan_by_id(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves full details and audit results for a specific scan."""
    scan = (
        db.query(Scan)
        .options(joinedload(Scan.results), joinedload(Scan.officer))
        .filter(Scan.id == scan_id)
        .first()
    )
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan #{scan_id} not found",
        )

    # Permission check: officer can only see own scans unless admin
    if current_user.role != UserRole.ADMIN and scan.officer_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this scan record",
        )

    return scan


@router.patch("/{scan_id}/results/{result_id}/review", response_model=ScanResultResponse)
def review_scan_result(
    scan_id: int,
    result_id: int,
    payload: OfficerReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Allows the inspecting officer to review and override a flagged declaration
    (e.g., confirm violation or mark as a false flag).
    """
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")

    if current_user.role != UserRole.ADMIN and scan.officer_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied")

    result = (
        db.query(ScanResult)
        .filter(ScanResult.id == result_id, ScanResult.scan_id == scan_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan result not found")

    result.officer_override = payload.officer_override
    db.commit()
    db.refresh(result)
    return result


@router.patch("/{scan_id}/results/{result_id}/confirm", response_model=ScanResultResponse)
def confirm_scan_result(
    scan_id: int,
    result_id: int,
    payload: OfficerConfirmRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    OCR-Failure Officer Confirmation Endpoint.

    Resolves a field that is in 'pending_user_confirmation' status. Three decisions:
    - "confirm_missing"          → Field genuinely absent → status becomes NON_COMPLIANT
    - "field_is_present"         → OCR missed it, officer confirms it exists → COMPLIANT
                                   Logged to ocr_corrections table for model improvement
    - "not_applicable_override"  → Officer determines this field doesn't apply → NOT_REQUIRED
                                   Logged to ocr_corrections table with reason

    After each confirmation, if no pending_user_confirmation fields remain, scan is
    automatically moved to COMPLETED status.
    """
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")

    if current_user.role != UserRole.ADMIN and scan.officer_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied")

    result = (
        db.query(ScanResult)
        .filter(ScanResult.id == result_id, ScanResult.scan_id == scan_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan result not found")

    decision = payload.user_decision

    if decision == "confirm_missing":
        result.compliance_status = ComplianceStatus.NON_COMPLIANT
        result.officer_override = OfficerOverride.CONFIRM_MISSING
        result.pending_reason = "Officer confirmed: field is genuinely absent from packaging."

    elif decision == "field_is_present":
        result.compliance_status = ComplianceStatus.COMPLIANT
        result.officer_override = OfficerOverride.FIELD_IS_PRESENT
        result.pending_reason = "Officer confirmed: field is present — OCR missed it."
        # Log to ocr_corrections for model improvement
        correction = OcrCorrection(
            scan_id=scan_id,
            result_id=result_id,
            field_name=result.field_name,
            officer_decision="field_is_present",
            override_reason=payload.override_reason,
            image_region=result.bounding_box,
        )
        db.add(correction)

    elif decision == "not_applicable_override":
        result.compliance_status = ComplianceStatus.NOT_REQUIRED
        result.officer_override = OfficerOverride.NOT_APPLICABLE_OVERRIDE
        result.pending_reason = (
            f"Officer override: field not applicable for this product. "
            + (f"Reason: {payload.override_reason}" if payload.override_reason else "")
        )
        # Log override
        correction = OcrCorrection(
            scan_id=scan_id,
            result_id=result_id,
            field_name=result.field_name,
            officer_decision="not_applicable_override",
            override_reason=payload.override_reason,
            image_region=result.bounding_box,
        )
        db.add(correction)

    db.commit()

    # Auto-finalize scan if no pending_user_confirmation fields remain
    remaining_pending = (
        db.query(ScanResult)
        .filter(
            ScanResult.scan_id == scan_id,
            ScanResult.compliance_status == ComplianceStatus.PENDING_USER_CONFIRMATION,
        )
        .count()
    )
    if remaining_pending == 0 and scan.status == ScanStatus.AWAITING_REVIEW:
        scan.status = ScanStatus.COMPLETED
        db.commit()

    db.refresh(result)
    return result


@router.patch("/{scan_id}/finalize", response_model=ScanResponse)
def finalize_scan(
    scan_id: int,
    payload: ScanFinalizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Finalize the overall scan with a shop name and final decision.
    """
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")
        
    if current_user.role != UserRole.ADMIN and scan.officer_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied")
        
    if payload.shop_name:
        scan.shop_name = payload.shop_name
    scan.final_decision = payload.final_decision
    scan.status = ScanStatus.COMPLETED
    
    db.commit()
    db.refresh(scan)
    return ScanResponse.from_orm_with_counts(scan)


@router.get("/{scan_id}/report")
def download_pdf_report(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generates and downloads a formal Legal Metrology Inspection Report in PDF format."""
    scan = (
        db.query(Scan)
        .options(joinedload(Scan.results), joinedload(Scan.officer))
        .filter(Scan.id == scan_id)
        .first()
    )
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")

    if current_user.role != UserRole.ADMIN and scan.officer_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied")

    pdf_path = generate_pdf_report(scan)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"Legal_Metrology_Inspection_{scan.id:06d}.pdf",
    )


@router.get("/{scan_id}/report/docx")
def download_docx_report(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generates and downloads a formal Legal Metrology Inspection Report in DOCX format."""
    scan = (
        db.query(Scan)
        .options(joinedload(Scan.results), joinedload(Scan.officer))
        .filter(Scan.id == scan_id)
        .first()
    )
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")

    if current_user.role != UserRole.ADMIN and scan.officer_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied")

    docx_path = generate_docx_report(scan)
    return FileResponse(
        docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"Legal_Metrology_Inspection_{scan.id:06d}.docx",
    )


@router.delete("/{scan_id}", status_code=status.HTTP_200_OK)
def delete_scan(
    scan_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deletes a scan report and all associated records from the database."""
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")

    # Clean up associated OCR corrections
    db.query(OcrCorrection).filter(OcrCorrection.scan_id == scan_id).delete()

    # Delete uploaded image files if present
    for url in scan.image_urls:
        fn = os.path.basename(url)
        fpath = UPLOAD_DIR / fn
        if fpath.exists():
            try:
                os.remove(fpath)
            except OSError:
                pass

    db.delete(scan)
    db.commit()
    return {"message": f"Scan #{scan_id} successfully deleted", "id": scan_id}


class BatchDeleteRequest(BaseModel):
    scan_ids: List[int]


@router.post("/batch-delete", status_code=status.HTTP_200_OK)
def batch_delete_scans(
    req: BatchDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deletes multiple scan reports and associated records from database."""
    if not req.scan_ids:
        return {"message": "No scans provided", "deleted_count": 0}

    scans = db.query(Scan).filter(Scan.id.in_(req.scan_ids)).all()

    deleted_ids = []
    for s in scans:
        deleted_ids.append(s.id)
        db.query(OcrCorrection).filter(OcrCorrection.scan_id == s.id).delete()
        for url in s.image_urls:
            fn = os.path.basename(url)
            fpath = UPLOAD_DIR / fn
            if fpath.exists():
                try:
                    os.remove(fpath)
                except OSError:
                    pass
        db.delete(s)

    db.commit()
    return {"message": f"Successfully deleted {len(deleted_ids)} reports", "deleted_ids": deleted_ids}


