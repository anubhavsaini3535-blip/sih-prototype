import sys
from sqlalchemy import text

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import SessionLocal, engine, Base
from app.models.user import User, UserRole
from app.models.rulebook import Rulebook
from app.models import OcrCorrection  # ensure table registered
from app.services.auth_service import hash_password

# ── Schema Migration (drop & recreate affected tables only) ──────────────────
# SQLite does not support ALTER COLUMN for Enum changes; we drop and recreate
# rulebooks and scan_results to apply the new schema.
# Users and scans tables are PRESERVED.
print("[MIGRATION] Dropping and recreating rulebooks, scan_results, ocr_corrections...")
with engine.connect() as conn:
    conn.execute(text("PRAGMA foreign_keys = OFF"))
    conn.execute(text("DROP TABLE IF EXISTS ocr_corrections"))
    conn.execute(text("DROP TABLE IF EXISTS scan_results"))
    conn.execute(text("DROP TABLE IF EXISTS rulebooks"))
    conn.execute(text("PRAGMA foreign_keys = ON"))
    conn.commit()

Base.metadata.create_all(bind=engine)
print("[MIGRATION] Schema updated successfully.")


def seed_database():
    db = SessionLocal()
    try:
        print("[INFO] Seeding Legal Metrology Compliance Checking System Database...")

        # ── Seed Users ────────────────────────────────────────────────────────
        admin_email = "admin@legalmetrology.gov.in"
        admin_user = db.query(User).filter(User.email == admin_email).first()
        if not admin_user:
            db.add(User(
                name="Chief Enforcement Admin",
                email=admin_email,
                hashed_password=hash_password("Admin@12345"),
                role=UserRole.ADMIN,
                department="Legal Metrology Central Headquarters",
            ))
            print(f"  [OK] Created Admin: {admin_email} (Password: Admin@12345)")
        else:
            admin_user.hashed_password = hash_password("Admin@12345")

        officer_email = "officer@legalmetrology.gov.in"
        officer_user = db.query(User).filter(User.email == officer_email).first()
        if not officer_user:
            db.add(User(
                name="Field Inspector Sharma",
                email=officer_email,
                hashed_password=hash_password("Officer@12345"),
                role=UserRole.OFFICER,
                department="Delhi Enforcement Wing",
            ))
            print(f"  [OK] Created Officer: {officer_email} (Password: Officer@12345)")
        else:
            officer_user.hashed_password = hash_password("Officer@12345")
        db.commit()

        # ── Validation Regexes ────────────────────────────────────────────────
        mrp_rx      = r"(?:Rs\.?|INR|MRP)\s*[\d,]+(?:\.\d{1,2})?"
        qty_rx      = r"\d+(?:\.\d+)?\s*(?:kg|g|gm|gms|ml|l|ltr|mg|pcs|units?|N|tablets?|capsules?|sheets?|nos?)(?:\s|$)"
        date_rx     = r"(?:0?[1-9]|1[0-2])[\/\-\.](?:20\d{2}|\d{2})|[A-Za-z]{3,9}[\s\/\-\.]\d{2,4}"
        expiry_rx   = r"\d{1,2}\s*months?|" + date_rx
        name_rx     = r".{3,}"
        origin_rx   = r"[A-Za-z][\w\s]{2,24}"
        fssai_rx    = r"\d{14}"
        drug_lic_rx = r"[A-Z]{1,3}[\d\/\-]{5,20}"
        phone_rx    = r"(?:\+91|0)[\s\-]?\d{4,5}[\s\-]?\d{5,6}|1[89]00[\s\-]?\d{6,8}"
        batch_rx    = r"[A-Z0-9\/\-]{3,20}"
        model_rx    = r"[A-Z0-9\/\-\s]{2,30}"
        power_rx    = r"\d+(?:\.\d+)?\s*(?:W|V|kW|VA|Hz|A)"

        # ── Category Definitions ─────────────────────────────────────────────
        # Format: (field_name, requirement_type, condition_description, is_mandatory, validation_regex)
        CATEGORIES = {

            "general_retail": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("net_quantity",              "mandatory",    None, True,  qty_rx),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("mfg_date",                  "mandatory",    None, True,  date_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("country_of_origin",         "conditional",  "Mandatory only for imported goods", False, origin_rx),
                ("dimensions",                "conditional",  "Applicable to size-relevant products only", False, name_rx),
                ("generic_name",              "mandatory",    None, True,  name_rx),
            ],

            "food_beverages": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("net_quantity",              "mandatory",    None, True,  qty_rx),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("mfg_date",                  "mandatory",    None, True,  date_rx),
                ("best_before_expiry",        "mandatory",    None, True,  expiry_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("country_of_origin",         "conditional",  "Mandatory only for imported goods", False, origin_rx),
                ("fssai_license_number",      "mandatory",    None, True,  fssai_rx),
                ("ingredients_list",          "mandatory",    None, True,  name_rx),
                ("veg_nonveg_symbol",         "mandatory",    None, True,  name_rx),
                ("allergen_info",             "conditional",  "Required if product contains common allergens (nuts, gluten, soy, dairy)", False, name_rx),
                ("batch_number",              "mandatory",    None, True,  batch_rx),
                ("generic_name",              "mandatory",    None, True,  name_rx),
            ],

            # Keep old "food" category for backward-compat with existing scan records
            "food": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("net_quantity",              "mandatory",    None, True,  qty_rx),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("mfg_date",                  "mandatory",    None, True,  date_rx),
                ("expiry_date",               "mandatory",    None, True,  expiry_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("country_of_origin",         "conditional",  "Mandatory only for imported goods", False, origin_rx),
                ("generic_name",              "mandatory",    None, True,  name_rx),
            ],

            "cosmetics": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("net_quantity",              "mandatory",    None, True,  qty_rx),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("mfg_date",                  "mandatory",    None, True,  date_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("country_of_origin",         "conditional",  "Mandatory only for imported goods", False, origin_rx),
                ("ingredients_list",          "mandatory",    None, True,  name_rx),
                ("usage_direction",           "conditional",  "Required for products with specific use instructions or safety warnings", False, name_rx),
                ("shelf_life_period",         "mandatory",    None, True,  expiry_rx),
                ("batch_number",              "mandatory",    None, True,  batch_rx),
                ("generic_name",              "mandatory",    None, True,  name_rx),
            ],

            "medicine": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("mfg_date",                  "mandatory",    None, True,  date_rx),
                ("expiry_date",               "mandatory",    None, True,  expiry_rx),
                ("batch_number",              "mandatory",    None, True,  batch_rx),
                ("composition",               "mandatory",    None, True,  name_rx),
                ("storage_instructions",      "mandatory",    None, True,  name_rx),
                ("drug_license_number",       "mandatory",    None, True,  drug_lic_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("net_quantity",              "conditional",  "Required for liquid or powder forms; not_applicable for strip/tablet-count packs", False, qty_rx),
                ("generic_name",              "mandatory",    None, True,  name_rx),
                # mfg_date is NOT not_applicable for all medicine; keep as mandatory above
                # net_quantity for strip medicines is not_applicable → seeded separately below
            ],

            "electronics": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("net_quantity",              "mandatory",    None, True,  qty_rx),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("mfg_date",                  "mandatory",    None, True,  date_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("country_of_origin",         "mandatory",    None, True,  origin_rx),  # Mandatory for electronics
                ("model_number",              "mandatory",    None, True,  model_rx),
                ("bis_certification_mark",    "conditional",  "Required for BIS-notified electronics categories (e.g. switches, cables, chargers)", False, name_rx),
                ("power_rating_voltage",      "mandatory",    None, True,  power_rx),
                ("warranty_period",           "conditional",  "Required when product carries manufacturer warranty", False, name_rx),
                ("generic_name",              "mandatory",    None, True,  name_rx),
            ],

            "clothes": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("fabric_composition",        "mandatory",    None, True,  name_rx),
                ("size",                      "mandatory",    None, True,  name_rx),
                ("care_instructions",         "conditional",  "Required for garments with specific washing, drying, or ironing requirements", False, name_rx),
                ("country_of_origin",         "conditional",  "Mandatory only for imported garments", False, origin_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("mfg_date",                  "not_applicable", "Manufacturing date is not required for clothing under Packaged Commodities Rules", False, None),
                ("net_quantity",              "not_applicable", "Net quantity by weight/volume not applicable to individual garments", False, None),
            ],

            "stationery": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("net_quantity",              "conditional",  "Required for countable or measurable stationery packs (e.g. 10 pens, 500 sheets)", False, qty_rx),
                ("age_recommendation",        "conditional",  "Required for children's stationery and art supplies intended for minors", False, name_rx),
                ("mfg_date",                  "conditional",  "Required only for consumable stationery (e.g. glue, ink, correction fluid)", False, date_rx),
                ("generic_name",              "mandatory",    None, True,  name_rx),
            ],

            # E-commerce listing: direct web scrape (excludes physical mfg_date)
            "ecommerce_listing": [
                ("manufacturer_name_address", "mandatory",    None, True,  None),
                ("net_quantity",              "mandatory",    None, True,  qty_rx),
                ("mrp",                       "mandatory",    None, True,  mrp_rx),
                ("consumer_care",             "mandatory",    None, True,  phone_rx),
                ("country_of_origin",         "mandatory",    None, True,  origin_rx),
                ("expiry_date",               "conditional",  "Required for perishable goods listings", False, expiry_rx),
                ("generic_name",              "mandatory",    None, True,  name_rx),
            ],
        }

        inserted = 0
        for cat, fields in CATEGORIES.items():
            for (field_name, req_type, cond_desc, is_mand, regex) in fields:
                existing = (
                    db.query(Rulebook)
                    .filter(Rulebook.category == cat, Rulebook.field_name == field_name)
                    .first()
                )
                if not existing:
                    rule = Rulebook(
                        category=cat,
                        field_name=field_name,
                        requirement_type=req_type,
                        condition_description=cond_desc,
                        is_mandatory=is_mand,
                        validation_regex=regex,
                    )
                    db.add(rule)
                    inserted += 1
                else:
                    existing.requirement_type = req_type
                    existing.condition_description = cond_desc
                    existing.is_mandatory = is_mand
                    existing.validation_regex = regex

        db.commit()
        cat_count = len(CATEGORIES)
        print(f"  [OK] Seeded {inserted} rules across {cat_count} categories: {', '.join(CATEGORIES.keys())}")
        print("[SUCCESS] Database seeding completed!\n")

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
