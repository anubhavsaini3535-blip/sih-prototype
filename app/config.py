import os
from pathlib import Path
import dotenv

# Load environment variables from .env file immediately
dotenv.load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
IS_VERCEL = bool(os.getenv("VERCEL"))

if IS_VERCEL:
    DATA_DIR = Path("/tmp/data")
    UPLOAD_DIR = Path("/tmp/uploads")
    REPORTS_DIR = Path("/tmp/reports")
else:
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = BASE_DIR / "uploads"
    REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")

# Security & Auth
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "legal-metrology-compliance-secret-key-32-chars-long-secure!!"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))

# Pipeline Thresholds
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 80.0))
FONT_SIZE_RATIO_THRESHOLD = float(os.getenv("FONT_SIZE_RATIO_THRESHOLD", 0.25))  # Lenient font size threshold
CONFIDENCE_PASS_THRESHOLD = float(os.getenv("CONFIDENCE_PASS_THRESHOLD", 0.80))
CONFIDENCE_REVIEW_THRESHOLD = float(os.getenv("CONFIDENCE_REVIEW_THRESHOLD", 0.40))
OVERALL_OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OVERALL_OCR_CONFIDENCE_THRESHOLD", 0.60))

# Field Matching Settings
FUZZY_MATCH_THRESHOLD = float(os.getenv("FUZZY_MATCH_THRESHOLD", 85.0))
AI_MATCH_LOG_FILE = DATA_DIR / "ai_resolved_matches.jsonl"
MAX_SCAN_IMAGES = int(os.getenv("MAX_SCAN_IMAGES", "3"))

# Gemini Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    import logging
    logging.warning("GEMINI_API_KEY is not set in environment variables. System will run in OCR-only mode (all low-confidence/missing fields will go to manual review).")

