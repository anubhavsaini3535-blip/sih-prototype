import os
import sys
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient

from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import app
from seed import seed_database

# Ensure database is seeded
seed_database()

client = TestClient(app)


def test_auth():
    print("\n--- 1. Testing Authentication & RBAC ---")

    # Login as Officer
    res = client.post(
        "/auth/login",
        data={"username": "officer@legalmetrology.gov.in", "password": "Officer@12345"},
    )
    assert res.status_code == 200, f"Officer login failed: {res.text}"
    officer_token = res.json()["access_token"]
    print("  [PASS] Officer login succeeded.")

    # Login as Admin
    res = client.post(
        "/auth/login",
        data={"username": "admin@legalmetrology.gov.in", "password": "Admin@12345"},
    )
    assert res.status_code == 200, f"Admin login failed: {res.text}"
    admin_token = res.json()["access_token"]
    print("  [PASS] Admin login succeeded.")

    # Register a new officer
    reg_data = {
        "name": "Inspector Verma",
        "email": "verma@legalmetrology.gov.in",
        "password": "Password@123",
        "role": "officer",
        "department": "Mumbai Field Unit",
    }
    res = client.post("/auth/register", json=reg_data)
    # Either 201 (created) or 400 (already exists)
    assert res.status_code in (201, 400), f"Registration failed: {res.text}"
    print("  [PASS] User registration validated.")

    return officer_token, admin_token


def test_blur_rejection(officer_token):
    print("\n--- 2. Testing Image Quality Check (Laplacian Blur Rejection) ---")
    # Generate intentionally blurry image
    blurry_img = np.full((300, 400, 3), 128, dtype=np.uint8)
    blurry_img = cv2.GaussianBlur(blurry_img, (51, 51), 0)
    _, encoded = cv2.imencode(".jpg", blurry_img)

    headers = {"Authorization": f"Bearer {officer_token}"}
    files = {"file": ("blurry_label.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")}
    data = {"product_category": "general_retail", "location": "Test Store"}

    res = client.post("/scans/upload", headers=headers, files=files, data=data)
    assert res.status_code == 400, f"Expected 400 blur rejection, got: {res.status_code} - {res.text}"
    detail = res.json().get("detail", "")
    assert "too blurry" in detail.lower() or "laplacian" in detail.lower()
    print(f"  [PASS] Blurry image rejected with message: {detail}")


def test_sharp_label_pipeline(officer_token):
    print("\n--- 3. Testing Full Pipeline (Physical Label Upload + OCR + Rule Check) ---")
    # Generate sharp synthetic product label with standard declarations
    img = np.full((400, 800, 3), 255, dtype=np.uint8)

    # Draw border
    cv2.rectangle(img, (10, 10), (790, 390), (0, 0, 0), 2)

    # Text declarations
    declarations = [
        ("Product Name: Nutritious Almond Pack", (30, 50), 0.8, (0, 0, 150), 2),
        ("MRP Rs. 299.00 (incl. of all taxes)", (30, 100), 0.8, (0, 0, 0), 2),
        ("Net Quantity: 500 g", (30, 150), 0.8, (0, 0, 0), 2),
        ("Mfg Date: 05/2026", (30, 200), 0.8, (0, 0, 0), 2),
        ("Country of Origin: India", (30, 250), 0.8, (0, 0, 0), 2),
        ("Mfg by: Royal Foods Pvt Ltd, Plot 12, Industrial Area, New Delhi", (30, 300), 0.65, (0, 0, 0), 2),
        ("Consumer Care: Call 1800-11-2233 or email care@royalfoods.com", (30, 350), 0.65, (0, 0, 0), 2),
    ]

    for text, pos, scale, color, thick in declarations:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    _, encoded = cv2.imencode(".png", img)

    headers = {"Authorization": f"Bearer {officer_token}"}
    files = {"file": ("packaged_almonds.png", io.BytesIO(encoded.tobytes()), "image/png")}
    data = {"product_category": "general_retail", "location": "Connaught Place Mart"}

    res = client.post("/scans/upload", headers=headers, files=files, data=data)
    assert res.status_code == 201, f"Upload scan failed: {res.status_code} - {res.text}"

    scan_data = res.json()
    scan_id = scan_data["id"]
    status_val = scan_data["status"]
    results = scan_data["results"]

    print(f"  [PASS] Scan created: ID #{scan_id}, Status: {status_val}")
    print(f"  [INFO] Extracted {len(results)} compliance declarations:")
    for r in results:
        print(f"    - {r['field_name']}: '{r['extracted_value']}' | Status: {r['compliance_status']} | Conf: {r['confidence_score']:.2f}")

    assert len(results) > 0, "No results generated"
    return scan_id, results[0]["id"]


def test_review_override(officer_token, scan_id, result_id):
    print("\n--- 4. Testing Officer Confirmation Flow (pending_user_confirmation → non_compliant) ---")
    headers = {"Authorization": f"Bearer {officer_token}"}

    # Use the new /confirm endpoint with confirm_missing decision
    payload = {"user_decision": "confirm_missing"}
    res = client.patch(f"/scans/{scan_id}/results/{result_id}/confirm", headers=headers, json=payload)
    assert res.status_code == 200, f"Confirm failed: {res.text}"
    data = res.json()
    assert data["compliance_status"] == "non_compliant", f"Expected non_compliant, got {data['compliance_status']}"
    assert data["officer_override"] == "confirm_missing"
    print(f"  [PASS] Officer confirmed field missing: status={data['compliance_status']}, override={data['officer_override']}")

    # Test field_is_present decision on the second pending result
    pending_results = [r for r in client.get(f"/scans/{scan_id}", headers=headers).json()["results"]
                       if r["compliance_status"] == "pending_user_confirmation"]
    if pending_results:
        r2_id = pending_results[0]["id"]
        payload2 = {"user_decision": "field_is_present", "override_reason": "Officer visually verified field on label"}
        res2 = client.patch(f"/scans/{scan_id}/results/{r2_id}/confirm", headers=headers, json=payload2)
        assert res2.status_code == 200, f"field_is_present confirm failed: {res2.text}"
        assert res2.json()["compliance_status"] == "compliant"
        print(f"  [PASS] Officer confirmed field_is_present: status={res2.json()['compliance_status']}")

    # Resolve all remaining pending fields so scan becomes completed
    scan_resp = client.get(f"/scans/{scan_id}", headers=headers).json()
    remaining = [r for r in scan_resp["results"] if r["compliance_status"] == "pending_user_confirmation"]
    for rem in remaining:
        client.patch(f"/scans/{scan_id}/results/{rem['id']}/confirm",
                     headers=headers, json={"user_decision": "confirm_missing"})

    final_scan = client.get(f"/scans/{scan_id}", headers=headers).json()
    print(f"  [PASS] Scan status after all confirmations: {final_scan['status']}")


def test_from_url(officer_token):
    print("\n--- 4b. Testing E-Commerce Listing URL Scan (POST /scans/from-url) ---")
    headers = {"Authorization": f"Bearer {officer_token}"}
    import app.routers.scans as scans_module

    # Mock the HTTP scraper return so test does not depend on live external web availability
    orig_scraper = scans_module.scrape_ecommerce_page
    scans_module.scrape_ecommerce_page = lambda url: (
        "Product Name: Premium Basmati Rice 5kg\n"
        "MRP: Rs. 499.00 (inclusive of all taxes)\n"
        "Net Quantity: 5 kg\n"
        "Manufactured by: Kohinoor Grain Mills, GT Road, Amritsar, Punjab\n"
        "Consumer Care: Phone 1800-200-1122 or email feedback@kohinoor.com\n"
        "Country of Origin: India\n"
    )
    try:
        payload = {
            "url": "https://example-store.in/products/basmati-rice-5kg",
            "product_category": "food",
            "location": "Online Marketplace",
        }
        res = client.post("/scans/from-url", headers=headers, json=payload)
        assert res.status_code == 201, f"From-URL scan failed: {res.status_code} - {res.text}"
        data = res.json()
        assert data["source_type"] == "ecommerce_listing"
        print(f"  [PASS] E-Commerce listing scan created: ID #{data['id']}")
        for r in data["results"]:
            print(f"    - {r['field_name']}: '{r['extracted_value']}' | Status: {r['compliance_status']}")
    finally:
        scans_module.scrape_ecommerce_page = orig_scraper


def test_reports(officer_token, scan_id):
    print("\n--- 5. Testing PDF and DOCX Report Generation ---")
    headers = {"Authorization": f"Bearer {officer_token}"}

    # PDF Report
    res_pdf = client.get(f"/scans/{scan_id}/report", headers=headers)
    assert res_pdf.status_code == 200, f"PDF report failed: {res_pdf.text}"
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert res_pdf.content.startswith(b"%PDF-"), "Invalid PDF signature"
    print(f"  [PASS] PDF Report generated successfully ({len(res_pdf.content)} bytes).")

    # DOCX Report
    res_docx = client.get(f"/scans/{scan_id}/report/docx", headers=headers)
    assert res_docx.status_code == 200, f"DOCX report failed: {res_docx.text}"
    assert "openxmlformats" in res_docx.headers["content-type"]
    assert res_docx.content.startswith(b"PK"), "Invalid DOCX zip archive header"
    print(f"  [PASS] DOCX Report generated successfully ({len(res_docx.content)} bytes).")


def test_history(officer_token):
    print("\n--- 6. Testing Paginated Scan History ---")
    headers = {"Authorization": f"Bearer {officer_token}"}
    res = client.get("/scans/history?page=1&page_size=5", headers=headers)
    assert res.status_code == 200, f"History failed: {res.text}"
    data = res.json()
    assert "items" in data and "total" in data
    print(f"  [PASS] Scan history retrieved: {data['total']} total records, page {data['page']}/{data['total_pages']}.")


def test_dashboard_and_rbac(officer_token, admin_token):
    print("\n--- 7. Testing Dashboard Stats & Admin RBAC ---")
    # Officer attempting admin dashboard -> 403 Forbidden
    res_officer = client.get("/dashboard/stats", headers={"Authorization": f"Bearer {officer_token}"})
    assert res_officer.status_code == 403, f"Expected 403 for officer, got: {res_officer.status_code}"
    print("  [PASS] Non-admin access to /dashboard/stats correctly forbidden (HTTP 403).")

    # Admin accessing dashboard -> 200 OK with analytics
    res_admin = client.get("/dashboard/stats", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_admin.status_code == 200, f"Admin dashboard failed: {res_admin.text}"
    stats = res_admin.json()
    print("  [PASS] Admin dashboard statistics retrieved:")
    print(f"    - Total Scans: {stats['total_scans']}")
    print(f"    - Total Violations: {stats['total_violations']}")
    print(f"    - Overall Violation Rate: {stats['overall_violation_rate_pct']}%")
    print(f"    - Category Rates: {stats['category_violation_rates']}")
    print(f"    - Status Breakdown: {stats['status_breakdown']}")
    print(f"    - Compliance Breakdown: {stats['compliance_breakdown']}")
    print(f"    - 30-Day Trend Items: {len(stats['last_30_days_trend'])} days")


def test_3_layer_field_matching(admin_token):
    print("\n--- 8. Testing 3-Layer Field Matching Architecture ---")
    from app.services.field_matcher import field_matcher

    # Test Layer 1: Exact Synonym Dictionary Match
    m1 = field_matcher.match_text("Maximum Retail Price Rs 450", enable_layer3=False)
    assert m1 is not None, "Layer 1 failed to match synonym"
    assert m1.field_name == "mrp"
    assert m1.match_layer == "exact_synonym"
    print(f"  [PASS] Layer 1 (Exact/Synonym): '{m1.matched_synonym}' -> field '{m1.field_name}' (layer: {m1.match_layer})")

    # Test Layer 2: Fuzzy String Matching via RapidFuzz (>= 85%) for typos/near-misses
    m2 = field_matcher.match_text("Max Retall Prlce: 250", enable_layer3=False)
    assert m2 is not None, "Layer 2 failed on OCR typo"
    assert m2.field_name == "mrp"
    assert m2.match_layer == "fuzzy_matching"
    assert m2.similarity_score >= 85.0
    print(f"  [PASS] Layer 2 (Fuzzy RapidFuzz): 'Max Retall Prlce' -> field '{m2.field_name}' (score: {m2.similarity_score}%, layer: {m2.match_layer})")

    # Test Layer 2 on Net Quantity typo: "Nt Quantlty"
    m2_qty = field_matcher.match_text("Nt Quantlty: 1 kg", enable_layer3=False)
    assert m2_qty is not None
    assert m2_qty.field_name == "net_quantity"
    assert m2_qty.match_layer == "fuzzy_matching"
    assert m2_qty.similarity_score >= 85.0
    print(f"  [PASS] Layer 2 (Fuzzy RapidFuzz): 'Nt Quantlty' -> field '{m2_qty.field_name}' (score: {m2_qty.similarity_score}%, layer: {m2_qty.match_layer})")

    # Test Layer 2 on Mfg Date typo: "Mfe Date"
    m2_mfg = field_matcher.match_text("Mfe Date: 05/2026", enable_layer3=False)
    assert m2_mfg is not None
    assert m2_mfg.field_name == "mfg_date"
    assert m2_mfg.match_layer == "fuzzy_matching"
    assert m2_mfg.similarity_score >= 85.0
    print(f"  [PASS] Layer 2 (Fuzzy RapidFuzz): 'Mfe Date' -> field '{m2_mfg.field_name}' (score: {m2_mfg.similarity_score}%, layer: {m2_mfg.match_layer})")

    # Test Layer 3: Gemini API fallback simulation & logging
    novel_label_text = "Taxes and customer billing sum: Rs 199"
    # Verify Layer 1 & 2 return None for novel unseen phrase
    assert field_matcher.match_text(novel_label_text, enable_layer3=False) is None

    # Test separate AI match logging
    from app.services.gemini_service import log_ai_resolved_match
    log_ai_resolved_match(
        input_snippet=novel_label_text,
        field_name="mrp",
        confidence=0.92,
        reason="gemini-2.0-flash classified billing sum as MRP",
        scan_id=999,
    )
    print("  [PASS] Layer 3 AI match successfully written to persistent JSONL log.")

    # Test Admin endpoint GET /dashboard/ai-matches
    res_matches = client.get("/dashboard/ai-matches", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_matches.status_code == 200, f"Failed to get AI matches: {res_matches.text}"
    matches_data = res_matches.json()
    assert matches_data["count"] > 0
    logged_item = matches_data["matches"][-1]
    assert logged_item["classified_field"] == "mrp"
    assert logged_item["input_snippet"] == novel_label_text
    print(f"  [PASS] Admin GET /dashboard/ai-matches retrieved logged AI match for field '{logged_item['classified_field']}' from model '{logged_item.get('model', 'gemini-2.0-flash')}'.")


def test_spatial_table_extraction():
    print("\n--- 9. Testing Spatial 2-Column Table Extraction ---")
    from app.services.extraction_service import extract_from_blocks

    blocks = [
        {"text": "Product Name", "confidence": 0.99, "bounding_box": {"x_min": 50, "y_min": 100, "x_max": 200, "y_max": 130}, "height": 30.0},
        {"text": "Pilot Hi-Tecpoint V5 Pen", "confidence": 0.98, "bounding_box": {"x_min": 250, "y_min": 98, "x_max": 600, "y_max": 132}, "height": 34.0},
        {"text": "Maximum Retail Price", "confidence": 0.99, "bounding_box": {"x_min": 50, "y_min": 150, "x_max": 200, "y_max": 180}, "height": 30.0},
        {"text": "75/- Per 1N Pen", "confidence": 0.97, "bounding_box": {"x_min": 250, "y_min": 148, "x_max": 500, "y_max": 182}, "height": 34.0},
        {"text": "Package Quantity", "confidence": 0.99, "bounding_box": {"x_min": 50, "y_min": 200, "x_max": 200, "y_max": 230}, "height": 30.0},
        {"text": "3 Pcs Blue", "confidence": 0.99, "bounding_box": {"x_min": 250, "y_min": 198, "x_max": 400, "y_max": 232}, "height": 34.0},
        {"text": "Month and year of Manufacture", "confidence": 0.99, "bounding_box": {"x_min": 50, "y_min": 250, "x_max": 200, "y_max": 280}, "height": 30.0},
        {"text": "November 2025", "confidence": 0.99, "bounding_box": {"x_min": 250, "y_min": 248, "x_max": 450, "y_max": 282}, "height": 34.0},
    ]

    extracted = extract_from_blocks(blocks)
    assert extracted["generic_name"].value == "Pilot Hi-Tecpoint V5 Pen"
    assert "75" in extracted["mrp"].value
    assert "3 Pcs Blue" in extracted["net_quantity"].value
    assert "November 2025" in extracted["mfg_date"].value
    print("  [PASS] Spatial 2-column extraction correctly parsed true values instead of table headers!")


def test_consumer_care_contact_detection():
    print("\n--- 10. Testing Enhanced Consumer Care Toll-Free & Email Detection ---")
    from app.services.extraction_service import normalize_ocr_contact_text, extract_from_blocks

    # Test 1: OCR @ corruption normalization
    raw_corrupted = "Contact: care©royalfoods.com or support(a)royalfoods.in"
    normalized = normalize_ocr_contact_text(raw_corrupted)
    assert "care@royalfoods.com" in normalized
    assert "support@royalfoods.in" in normalized
    print(f"  [PASS] OCR @ symbol corruption normalization: '{raw_corrupted}' -> '{normalized}'")

    # Test 2: Extraction of Toll-Free number and email from packaging blocks
    blocks = [
        {"text": "Commodity: Wheat Flour 10kg", "confidence": 0.99, "bounding_box": {"x_min": 20, "y_min": 50, "x_max": 400, "y_max": 80}, "height": 25.0},
        {"text": "MRP Rs. 380.00", "confidence": 0.99, "bounding_box": {"x_min": 20, "y_min": 100, "x_max": 200, "y_max": 130}, "height": 25.0},
        {"text": "Net Qty: 10 kg", "confidence": 0.99, "bounding_box": {"x_min": 20, "y_min": 150, "x_max": 200, "y_max": 180}, "height": 25.0},
        {"text": "Consumer Care Executive: Quality Cell", "confidence": 0.99, "bounding_box": {"x_min": 20, "y_min": 200, "x_max": 450, "y_max": 230}, "height": 25.0},
        {"text": "Toll Free No: 1800-22-3344", "confidence": 0.98, "bounding_box": {"x_min": 20, "y_min": 240, "x_max": 350, "y_max": 270}, "height": 25.0},
        {"text": "Email: feedback©wheatflour.co.in", "confidence": 0.98, "bounding_box": {"x_min": 20, "y_min": 280, "x_max": 420, "y_max": 310}, "height": 25.0},
    ]

    res = extract_from_blocks(blocks)
    assert "consumer_care" in res, "Consumer care was not extracted"
    care_val = res["consumer_care"].value
    assert "1800-22-3344" in care_val, f"Toll free number missing: {care_val}"
    assert "feedback@wheatflour.co.in" in care_val, f"Normalized email missing: {care_val}"
    print(f"  [PASS] Consumer care combined toll-free & normalized email: '{care_val}'")

    # Test 3: Standalone un-labelled toll-free helpline detection
    blocks_unlabelled = [
        {"text": "Item: Hand Sanitizer", "confidence": 0.99, "bounding_box": None, "height": 20.0},
        {"text": "Net: 500 ml", "confidence": 0.99, "bounding_box": None, "height": 20.0},
        {"text": "MRP Rs. 150", "confidence": 0.99, "bounding_box": None, "height": 20.0},
        {"text": "For complaints call Toll-Free 1800-425-3242 or email customercare@clean.com", "confidence": 0.99, "bounding_box": None, "height": 20.0},
    ]
    res_unlabelled = extract_from_blocks(blocks_unlabelled)
    assert "consumer_care" in res_unlabelled
    assert "1800-425-3242" in res_unlabelled["consumer_care"].value
    assert "customercare@clean.com" in res_unlabelled["consumer_care"].value
    print("  [PASS] Standalone unlabelled consumer care toll-free/email auto-detected successfully!")


def test_product_by_and_domestic_origin():
    print("\n--- 11. Testing 'Product by' Manufacturer Extraction & Domestic Origin ---")
    from app.services.extraction_service import extract_from_blocks

    blocks = [
        {"text": "MRP", "confidence": 0.99, "bounding_box": {"x_min": 230, "y_min": 240, "x_max": 260, "y_max": 290}, "height": 25.0},
        {"text": "65.00", "confidence": 0.99, "bounding_box": {"x_min": 230, "y_min": 295, "x_max": 280, "y_max": 350}, "height": 30.0},
        {"text": "Quantity: 1 N", "confidence": 0.99, "bounding_box": {"x_min": 50, "y_min": 750, "x_max": 80, "y_max": 870}, "height": 25.0},
        {"text": "Long Book", "confidence": 0.99, "bounding_box": {"x_min": 680, "y_min": 20, "x_max": 820, "y_max": 50}, "height": 28.0},
        {"text": "A Quality Product by: NAVNEET EDUCATION LIMITED", "confidence": 0.95, "bounding_box": {"x_min": 15, "y_min": 120, "x_max": 490, "y_max": 150}, "height": 26.0},
        {"text": "Reg Off: Navneet Bhavan, Dadar (West), Mumbai 400 028, Maharashtra", "confidence": 0.99, "bounding_box": {"x_min": 15, "y_min": 155, "x_max": 870, "y_max": 180}, "height": 25.0},
        {"text": "Customer Care No.: 022 66626300, Email: stationery@navneet.com", "confidence": 0.99, "bounding_box": {"x_min": 15, "y_min": 205, "x_max": 750, "y_max": 230}, "height": 24.0},
        {"text": "8902442220232", "confidence": 0.99, "bounding_box": {"x_min": 20, "y_min": 90, "x_max": 220, "y_max": 110}, "height": 18.0},
    ]

    res = extract_from_blocks(blocks)
    assert "mrp" in res, "MRP missing"
    assert "65.00" in res["mrp"].value, f"Expected 65.00 in MRP, got {res['mrp'].value}"
    assert "manufacturer_details" in res, "Manufacturer details missing"
    assert "NAVNEET EDUCATION LIMITED" in res["manufacturer_details"].value, f"Company name missing: {res['manufacturer_details'].value}"
    assert "Mumbai 400 028" in res["manufacturer_details"].value, f"Registered office missing: {res['manufacturer_details'].value}"
    assert "generic_name" in res, "Generic name missing"
    assert "Long Book" in res["generic_name"].value, f"Expected Long Book as generic name, got {res['generic_name'].value}"
    assert "country_of_origin" in res, "Country of origin missing"
    assert res["country_of_origin"].value == "India", f"Expected India as country of origin, got {res['country_of_origin'].value}"
    print("  [PASS] 'A Quality Product by' mapped to manufacturer, generic name assigned to commodity, and domestic origin deduced as India.")


def test_multi_image_upload_and_limit(officer_token):
    print("\n--- 12. Testing Multi-Image Packaging Upload (Max 3) & Bypassed Ollama Speed ---")
    headers = {"Authorization": f"Bearer {officer_token}"}

    def make_sharp_card(text_lines):
        img = np.full((350, 700, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (5, 5), (695, 345), (0, 0, 0), 2)
        for idx, text in enumerate(text_lines):
            cv2.putText(img, text, (20, 60 + idx * 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        _, encoded = cv2.imencode(".jpg", img)
        return encoded.tobytes()

    p1_bytes = make_sharp_card(["Commodity: Organic Green Tea", "Net Quantity: 100 g", "MRP Rs. 240.00"])
    p2_bytes = make_sharp_card(["Mfg Date: 10/2026", "Country of Origin: India"])
    p3_bytes = make_sharp_card(["Marketed by: Green Pure Foods Ltd, Mumbai 400050", "Toll Free: 1800-22-9900"])
    p4_bytes = make_sharp_card(["Extra Panel Notes: Store in cool dry place"])

    # Test 1: Exceeding maximum 3 images limit
    files_4 = [
        ("files", ("panel1.jpg", io.BytesIO(p1_bytes), "image/jpeg")),
        ("files", ("panel2.jpg", io.BytesIO(p2_bytes), "image/jpeg")),
        ("files", ("panel3.jpg", io.BytesIO(p3_bytes), "image/jpeg")),
        ("files", ("panel4.jpg", io.BytesIO(p4_bytes), "image/jpeg")),
    ]
    res_reject = client.post(
        "/scans/upload",
        headers=headers,
        files=files_4,
        data={"product_category": "food", "location": "Retail Mart"},
    )
    assert res_reject.status_code == 400, f"Expected 400 limit rejection, got {res_reject.status_code} - {res_reject.text}"
    assert "maximum 3 images allowed" in res_reject.json()["detail"].lower()
    print(f"  [PASS] 4-panel scan rejected with message: '{res_reject.json()['detail']}'")

    # Test 2: Valid 2-image multi-panel packaging upload
    files_2 = [
        ("files", ("front_panel.jpg", io.BytesIO(p1_bytes), "image/jpeg")),
        ("files", ("back_panel.jpg", io.BytesIO(p2_bytes), "image/jpeg")),
    ]
    import time
    start_t = time.time()
    res_valid = client.post(
        "/scans/upload",
        headers=headers,
        files=files_2,
        data={"product_category": "food", "location": "Retail Mart"},
    )
    elapsed = time.time() - start_t
    assert res_valid.status_code == 201, f"Multi-image scan upload failed: {res_valid.status_code} - {res_valid.text}"
    scan_data = res_valid.json()
    assert len(scan_data["image_urls"]) == 2, f"Expected 2 image URLs, got {scan_data['image_urls']}"
    print(f"  [PASS] 2-panel scan processed successfully in {elapsed:.2f}s without Ollama timeout!")
    print(f"    - Image URLs saved: {scan_data['image_urls']}")

    # Verify field extraction across multiple panels
    results_map = {r["field_name"]: r for r in scan_data["results"]}
    assert "mrp" in results_map, "MRP not found across panels"
    assert "net_quantity" in results_map, "Net quantity not found across panels"
    print("    - Fields extracted across panels: " + ", ".join(results_map.keys()))

    # Test 3: PDF and DOCX reports work with multiple packaging images
    multi_id = scan_data["id"]
    res_pdf = client.get(f"/scans/{multi_id}/report", headers=headers)
    assert res_pdf.status_code == 200 and res_pdf.content.startswith(b"%PDF-")
    res_docx = client.get(f"/scans/{multi_id}/report/docx", headers=headers)
    assert res_docx.status_code == 200 and res_docx.content.startswith(b"PK")
    print("  [PASS] PDF & DOCX reports generated successfully with multi-panel packaging metadata.")


if __name__ == "__main__":
    print("[START] Running Legal Metrology Compliance Checking System Verification Tests...")
    officer_tok, admin_tok = test_auth()
    test_blur_rejection(officer_tok)
    scan_id, result_id = test_sharp_label_pipeline(officer_tok)
    test_review_override(officer_tok, scan_id, result_id)
    test_from_url(officer_tok)
    test_reports(officer_tok, scan_id)
    test_history(officer_tok)
    test_dashboard_and_rbac(officer_tok, admin_tok)
    test_3_layer_field_matching(admin_tok)
    test_spatial_table_extraction()
    test_consumer_care_contact_detection()
    test_product_by_and_domestic_origin()
    test_multi_image_upload_and_limit(officer_tok)
    print("\n[SUCCESS] ALL INTEGRATION & REGULATORY PIPELINE TESTS PASSED!")

